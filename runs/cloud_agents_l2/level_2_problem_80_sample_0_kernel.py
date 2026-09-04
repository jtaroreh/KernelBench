import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_gemm_kernel(
    x_ptr, out_ptr,
    batch_size, out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # This kernel performs:
    # 1. Max reduction along dim 1
    # 2. Mean reduction along dim 1
    # 3. Subtraction and GELU
    
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    
    # Load row and compute max/mean
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulators
    max_val = -float('inf')
    sum_val = 0.0
    
    # We need to compute max and mean across the row
    # Since we need the max value to subtract, we do two passes or load once
    # Given the constraints, we load the row into SRAM
    for i in range(0, out_features, BLOCK_SIZE_N):
        cols = i + col_offsets
        mask = (row_offsets[:, None] < batch_size) & (cols[None, :] < out_features)
        x = tl.load(x_ptr + row_offsets[:, None] * out_features + cols[None, :], mask=mask, other=-float('inf'))
        
        max_val = tl.maximum(max_val, tl.max(x, axis=1))
        sum_val += tl.sum(x, axis=1)
    
    mean_val = sum_val / out_features
    
    # Second pass to apply GELU
    for i in range(0, out_features, BLOCK_SIZE_N):
        cols = i + col_offsets
        mask = (row_offsets[:, None] < batch_size) & (cols[None, :] < out_features)
        
        # Compute: x = max_val - mean_val
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        val = max_val[:, None] - mean_val[:, None]
        
        # GELU approximation
        cdf = 0.5 * (1.0 + tl.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
        out = val * cdf
        
        tl.store(out_ptr + row_offsets[:, None] * 1 + 0, out, mask=mask[:, 0:1])

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.out_features = out_features

    def forward(self, x):
        x = self.gemm(x)
        batch_size = x.shape[0]
        out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = triton.next_power_of_2(self.out_features)
        
        grid = (triton.cdiv(batch_size, BLOCK_SIZE_M),)
        
        fused_post_gemm_kernel[grid](
            x, out,
            batch_size, self.out_features,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        return out