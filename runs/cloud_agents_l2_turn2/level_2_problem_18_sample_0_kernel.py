import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_outm,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    m = tl.program_id(0)
    
    # Accumulator for the final sum for this row 'm'
    row_sum_acc = 0.0
    
    # We need to compute (X @ W.T + B).sum(dim=1)
    # This is equivalent to (X @ W.T).sum(dim=1) + B.sum()
    # Since we need the sum of the whole row, we can accumulate partial sums.
    
    for n_block_start in tl.range(0, N, BLOCK_SIZE_N):
        n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N
        
        # Accumulator for the current block of N
        partial_n_acc = 0.0
        
        for k_block_start in tl.range(0, K, BLOCK_SIZE_K):
            k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < K
            
            x_block = tl.load(x_ptr + m * stride_xm + k_offsets * stride_xk, mask=k_mask, other=0.0)
            w_block = tl.load(w_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk,
                              mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            
            # x_block is (BLOCK_SIZE_K,), w_block is (BLOCK_SIZE_N, BLOCK_SIZE_K)
            # We want (x_block @ w_block.T) which is (1, BLOCK_SIZE_N)
            # tl.dot expects 2D tensors. Reshape x_block to (1, BLOCK_SIZE_K)
            partial_n_acc += tl.sum(tl.dot(x_block[None, :], tl.trans(w_block)), axis=0)
            
        bias_block = tl.load(b_ptr + n_offsets, mask=n_mask, other=0.0)
        row_sum_acc += tl.sum(partial_n_acc + bias_block)
        
    tl.store(out_ptr + m * stride_outm, row_sum_acc)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features
        
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        
        grid = (M,)
        BLOCK_SIZE_K = 128
        BLOCK_SIZE_N = 128
        
        linear_sum_kernel[grid](
            x, self.linear.weight, self.linear.bias, out,
            M, K, N,
            x.stride(0), x.stride(1),
            self.linear.weight.stride(0), self.linear.weight.stride(1),
            out.stride(0),
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        return out