import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_bias_relu_kernel(
    A, B, Bias, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    Bias_ptr = Bias + rn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr)
    acc += bias[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)

    out_ptr = Out + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    tl.store(out_ptr, acc)

def triton_gemm_bias_relu(x, weight, bias):
    M, K = x.shape
    K, N = weight.shape
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    gemm_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Transpose weight to match standard GEMM (M,K) @ (K,N)
        return triton_gemm_bias_relu(x, self.weight, self.bias)