import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Define the fused Triton kernel
@triton.jit
def fused_linear_scalar_mul_leaky_relu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B is (N, K) for nn.Linear, so stride_bn for N, stride_bk for K
    stride_cm, stride_cn,
    multiplier, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    This kernel performs a fused operation:
    C = LeakyReLU( (A @ B.T + bias) * multiplier )

    A: Input tensor (M, K)
    B: Weight tensor (N, K) - will be transposed to (K, N) for matmul
    bias: Bias tensor (N,)
    C: Output tensor (M, N)
    """
    # Map program id to the block of output C(M, N)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Pointers for A (input x, shape M x K)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_ak = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)

    # Pointers for B (weight W, shape N x K)
    # We load a block of B of shape (BLOCK_N, BLOCK_K)
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_bk = tl.arange(0, BLOCK_K)
    b_ptrs = B_ptr + (offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)

    # Pointers for bias (shape N)
    bias_ptrs = bias_ptr + offs_bn

    # Accumulator for C (output, shape M x N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load a block of A (BLOCK_M x BLOCK_K)
        a = tl.load(a_ptrs + k * stride_ak,
                    mask=(offs_am[:, None] < M) & ((k + offs_ak[None, :]) < K),
                    other=0.0)
        # Load a block of B (BLOCK_N x BLOCK_K)
        b = tl.load(b_ptrs + k * stride_bk,
                    mask=(offs_bn[:, None] < N) & ((k + offs_bk[None, :]) < K),
                    other=0.0)
        # Perform matrix multiplication: A (BLOCK_M x BLOCK_K) @ B.T (BLOCK_K x BLOCK_N)
        accumulator += tl.dot(a, b, trans_b=True)

    # Load bias (BLOCK_N,)
    bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0)
    # Add bias to each row of the accumulator (BLOCK_M x BLOCK_N)
    accumulator += bias[None, :]

    # Apply scalar multiplication
    accumulator *= multiplier

    # Apply LeakyReLU
    accumulator = tl.where(accumulator >= 0, accumulator, accumulator * negative_slope)

    # Store C
    offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator,
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def triton_fused_linear_scalar_mul_leaky_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: float,
    negative_slope: float
):
    """
    Wrapper function to launch the fused Triton kernel.
    Performs: LeakyReLU( (x @ weight.T + bias) * multiplier )
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32, "Tensors must be FP32."

    # Ensure inputs are contiguous for optimal memory access patterns
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Dimensions
    M, K = x.shape
    N, K_weight = weight.shape
    assert K == K_weight, f"Input K ({K}) must match weight K ({K_weight})"
    assert bias.shape[0] == N, f"Bias N ({bias.shape[0]}) must match weight N ({N})"

    # Output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Strides for input, weight, and output tensors
    stride_am, stride_ak = x.stride()
    stride_bn, stride_bk = weight.stride() # weight is (N, K), so stride_bn for N, stride_bk for K
    stride_cm, stride_cn = out.stride()

    # Triton tuning parameters (can be optimized further for specific hardware/workloads)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8 # Heuristic for grouping blocks along M dimension to improve cache locality

    # Grid dimensions for the kernel launch
    # Each program instance computes a BLOCK_M x BLOCK_N tile of the output C
    grid = lambda META: (
        tl.cdiv(M, META['BLOCK_M']) * tl.cdiv(N, META['BLOCK_N']),
    )

    # Launch the Triton kernel
    fused_linear_scalar_mul_leaky_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        multiplier, negative_slope,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces the Gemm, scalar multiplication, and LeakyReLU
    with a single fused Triton kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # We still use nn.Linear to manage the weight and bias parameters,
        # but its forward method will not be called directly.
        # Ensure weight and bias are on CUDA and FP32.
        self.linear_layer = nn.Linear(in_features, out_features)
        self.multiplier_val = multiplier
        self.negative_slope_val = negative_slope

    def forward(self, x):
        # Call our custom Triton-based fused operation
        return triton_fused_linear_scalar_mul_leaky_relu(
            x,
            self.linear_layer.weight,
            self.linear_layer.bias,
            self.multiplier_val,
            self.negative_slope_val
        )