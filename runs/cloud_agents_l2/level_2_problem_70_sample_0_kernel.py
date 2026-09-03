import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_sigmoid_scale_add_kernel(
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
    
    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rn[None, :] * stride_wn + rk[:, None] * stride_wk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=rk[None, :] < K - k, other=0.0)
        w = tl.load(w_ptrs, mask=rk[:, None] < K - k, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        
    bias = tl.load(b_ptr + rn)
    acc += bias[None, :]
    
    # Fusion: Sigmoid, Scaling, and Residual Add
    # original_x is the result of the GEMM (acc)
    original_x = acc
    sigmoid_x = tl.sigmoid(acc)
    scaled_x = sigmoid_x * scaling_factor
    out = scaled_x + original_x
    
    out_ptrs = out_ptr + (rm[:, None] * N + rn[None, :])
    tl.store(out_ptrs, out)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = float(scaling_factor)
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        M, K = x.shape
        N = self.hidden_size
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        gemm_sigmoid_scale_add_kernel[grid](
            x, self.weight, self.bias, out,
            M, N, K,
            self.scaling_factor,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        return out