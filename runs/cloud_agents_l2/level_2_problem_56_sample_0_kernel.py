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
    # Grid: (M / BLOCK_SIZE_M)
    pid = tl.program_id(0)
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Accumulator for the sum
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load input block
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk, mask=rk[None, :] < K, other=0.0)
        
        # Iterate over N dimension to compute linear + sigmoid
        for n in range(0, N, BLOCK_SIZE_N):
            rn = n + tl.arange(0, BLOCK_SIZE_N)
            
            # Load weights
            w = tl.load(w_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk, mask=rk[:, None] < K, other=0.0)
            
            # Matmul partial
            linear = tl.dot(x, w)
            
            # Add bias if exists
            if b_ptr is not None:
                bias = tl.load(b_ptr + rn)
                linear += bias[None, :]
            
            # Sigmoid
            sig = tl.sigmoid(linear)
            
            # Sum over N
            acc += tl.sum(sig, axis=1)

    tl.store(out_ptr + rm, acc)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size = x.shape[0]
        out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        
        w = self.linear.weight
        b = self.linear.bias
        
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 64
        
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