import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_relu_div_kernel(
    A, B, Out,
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

    # Apply ReLU, divide by constant, and store
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc / divisor
    
    out_ptr = Out + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        M, K = x.shape
        N = self.linear.weight.shape[0]
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.linear.weight.t().contiguous()
        bias = self.linear.bias
        
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        matmul_relu_div_kernel[grid](
            x, weight, out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1),
            self.divisor,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        
        # Add bias separately as it's a simple broadcast add
        if bias is not None:
            out += bias / self.divisor
            
        return out