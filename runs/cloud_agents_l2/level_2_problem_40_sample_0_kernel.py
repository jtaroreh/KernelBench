import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_scale_add_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    scaling_factor,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk_curr = k + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk_curr[None, :] * stride_xk, mask=rk_curr[None, :] < K, other=0.0)
        w = tl.load(w_ptr + rk_curr[:, None] * stride_wk + rn[None, :] * stride_wn, mask=rk_curr[:, None] < K, other=0.0)
        acc += tl.dot(x, w)

    # Add bias
    bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc += bias[None, :]

    # Apply scaling and residual: x = (x * scaling_factor) + x
    # Since x is the result of matmul+bias, we compute (acc * scaling_factor) + acc
    res = acc * (scaling_factor + 1.0)

    tl.store(out_ptr + rm[:, None] * N + rn[None, :], res, mask=(rm[:, None] < M) & (rn[None, :] < N))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        matmul_scale_add_kernel[grid](
            x, self.linear.weight.t(), self.linear.bias, out,
            M, N, K,
            self.scaling_factor,
            x.stride(0), x.stride(1),
            self.linear.weight.stride(0), self.linear.weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        return out