import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_div_gelu_kernel(
    A, B, Bias, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    divisor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_SIZE_K * stride_ak
        B_ptr += BLOCK_SIZE_K * stride_bk

    if Bias is not None:
        bias = tl.load(Bias + rn)
        acc += bias[None, :]

    # Apply division and GELU
    acc = acc / divisor
    # GELU approximation using tl.math.tanh
    cdf = 0.5 * (1.0 + tl.math.tanh(0.79788456 * (acc + 0.044715 * acc * acc * acc)))
    acc = acc * cdf

    tl.store(Out + (rm[:, None] * stride_om + rn[None, :] * stride_on), acc)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        M, K = x.shape
        N = self.linear.weight.shape[0]
        
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        # Grid configuration
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        matmul_div_gelu_kernel[grid](
            x, self.linear.weight.t(), self.linear.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.linear.weight.stride(0), self.linear.weight.stride(1),
            out.stride(0), out.stride(1),
            self.divisor,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        return out