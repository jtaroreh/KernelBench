import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def dropout_softmax_kernel(
    x_ptr, out_ptr,
    stride_xm, stride_xn,
    n_cols,
    p, seed,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map program to row
    row_idx = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Load row data
    mask_m = row_idx < 128 # batch_size
    
    # Online Softmax variables
    m_i = tl.full([BLOCK_SIZE_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # We need to process columns in blocks
    for start_n in range(0, n_cols, BLOCK_SIZE_N):
        cols = start_n + col_offsets
        mask_n = cols < n_cols
        
        # Load input
        x = tl.load(x_ptr + row_idx[:, None] * stride_xm + cols[None, :] * stride_xn, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        
        # Apply Dropout
        # Generate random mask using Philox-like approach or simple hash
        # For simplicity in this kernel, we use a threshold check
        rand = tl.rand(seed, row_idx[:, None] * n_cols + cols[None, :])
        x = tl.where(rand > p, x / (1.0 - p), 0.0)
        
        # Softmax update
        m_i_new = tl.maximum(m_i, tl.max(x, 1))
        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + tl.sum(tl.exp(x - m_i_new[:, None]), 1)
        m_i = m_i_new
        
    # Final normalization
    for start_n in range(0, n_cols, BLOCK_SIZE_N):
        cols = start_n + col_offsets
        mask_n = cols < n_cols
        x = tl.load(x_ptr + row_idx[:, None] * stride_xm + cols[None, :] * stride_xn, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        rand = tl.rand(seed, row_idx[:, None] * n_cols + cols[None, :])
        x = tl.where(rand > p, x / (1.0 - p), 0.0)
        
        out = tl.exp(x - m_i[:, None]) / l_i[:, None]
        tl.store(out_ptr + row_idx[:, None] * stride_xm + cols[None, :] * stride_xn, out, mask=mask_m[:, None] & mask_n[None, :])

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.p = dropout_p

    def forward(self, x):
        x = self.linear(x)
        out = torch.empty_like(x)
        
        # Kernel configuration
        BLOCK_SIZE_N = 1024
        grid = (triton.cdiv(x.shape[0], 1),)
        
        dropout_softmax_kernel[grid](
            x, out,
            x.stride(0), x.stride(1),
            x.shape[1],
            self.p, 42,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        return out