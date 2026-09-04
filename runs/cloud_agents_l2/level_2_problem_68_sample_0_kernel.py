import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_min_sub_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    constant,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rn[None, :] * stride_wn + rk[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs + k * stride_xk, mask=rk[None, :] < K - k, other=0.0)
        w = tl.load(w_ptrs + k * stride_wk, mask=rk[:, None] < K - k, other=0.0)
        acc += tl.dot(x, w)

    if b_ptr is not None:
        bias = tl.load(b_ptr + rn)
        acc += bias[None, :]

    # Fused min and sub: min(x, constant) - constant
    out = tl.minimum(acc, constant) - constant

    out_ptrs = out_ptr + (rm[:, None] * N + rn[None, :])
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = constant
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        M, K = x.shape
        N = self.out_features
        
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        linear_min_sub_kernel[grid](
            x, self.linear.weight.T, self.linear.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.linear.weight.stride(0), self.linear.weight.stride(1),
            self.constant,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        return out