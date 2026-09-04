import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gelu_softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map row index
    row_idx = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    row_mask = row_idx[:, None] < n_rows
    
    # Online Softmax: find max for stability
    m_i = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], float("-inf"), dtype=tl.float32)
    
    # First pass: GELU and find max
    for start_col in range(0, n_cols, BLOCK_SIZE_N):
        cols = start_col + col_offsets
        mask = (cols[None, :] < n_cols) & row_mask
        x = tl.load(x_ptr + row_idx[:, None] * n_cols + cols[None, :], mask=mask, other=float("-inf"))
        
        # GELU using tl.math.tanh
        cdf = 0.5 * (1.0 + tl.math.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
        x = x * cdf
        
        m_i = tl.maximum(m_i, tl.where(mask, x, float("-inf")))
        
    # Second pass: exp and sum
    l_i = tl.zeros([BLOCK_SIZE_M, 1], dtype=tl.float32)
    for start_col in range(0, n_cols, BLOCK_SIZE_N):
        cols = start_col + col_offsets
        mask = (cols[None, :] < n_cols) & row_mask
        x = tl.load(x_ptr + row_idx[:, None] * n_cols + cols[None, :], mask=mask, other=float("-inf"))
        
        cdf = 0.5 * (1.0 + tl.math.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
        x = x * cdf
        
        # Prevent overflow in exp
        x_clamped = tl.clamp(x - m_i, -88.0, 88.0)
        p = tl.exp(x_clamped)
        tl.store(out_ptr + row_idx[:, None] * n_cols + cols[None, :], p, mask=mask)
        l_i += tl.sum(tl.where(mask, p, 0.0), axis=1)[:, None]

    # Normalize
    for start_col in range(0, n_cols, BLOCK_SIZE_N):
        cols = start_col + col_offsets
        mask = (cols[None, :] < n_cols) & row_mask
        p = tl.load(out_ptr + row_idx[:, None] * n_cols + cols[None, :], mask=mask, other=0.0)
        tl.store(out_ptr + row_idx[:, None] * n_cols + cols[None, :], p / l_i, mask=mask)

def triton_gelu_softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = triton.next_power_of_2(n_cols)
    
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    gelu_softmax_kernel[grid](x, out, n_cols, n_rows, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out

class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax using Triton.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return triton_gelu_softmax(x)