import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Accumulator for the sum over N
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # We need to compute (X @ W.T + B) and then sum sigmoid(result) over N.
    # Since N is large (32768), we iterate over N blocks.
    for n in range(0, N, BLOCK_SIZE_N):
        rn = n + tl.arange(0, BLOCK_SIZE_N)
        
        # Accumulator for the linear part (M, BLOCK_SIZE_N)
        linear = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        
        # Matmul: X @ W.T
        for k in range(0, K, BLOCK_SIZE_K):
            rk = k + tl.arange(0, BLOCK_SIZE_K)
            
            x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk, mask=rm[:, None] < M, other=0.0)
            w = tl.load(w_ptr + rn[None, :] * stride_wk + rk[:, None] * stride_wn, mask=rn[None, :] < N, other=0.0)
            
            linear += tl.dot(x, w)
            
        # Add bias
        if b_ptr is not None:
            bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
            linear += bias[None, :]
            
        # Sigmoid
        sig = tl.sigmoid(linear)
        
        # Sum over N
        acc += tl.sum(sig, axis=1)

    # Store result
    tl.store(out_ptr + rm[:, None], acc[:, None], mask=rm[:, None] < M)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, 1), device=x.device, dtype=torch.float32)
        
        w = self.linear.weight
        b = self.linear.bias
        
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 128
        
        grid = (triton.cdiv(batch_size, BLOCK_SIZE_M),)
        
        linear_sigmoid_sum_kernel[grid](
            x, w, b, out,
            batch_size, self.hidden_size, self.input_size,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        return out