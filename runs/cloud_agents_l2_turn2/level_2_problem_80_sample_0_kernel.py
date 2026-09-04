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
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulators as tensors of shape (BLOCK_SIZE_M,)
    max_val = tl.full([BLOCK_SIZE_M], -float('inf'), dtype=tl.float32)
    sum_val = tl.full([BLOCK_SIZE_M], 0.0, dtype=tl.float32)
    
    # First pass: compute max and sum
    for i in range(0, out_features, BLOCK_SIZE_N):
        col_offsets = i + tl.arange(0, BLOCK_SIZE_N)
        mask = (row_offsets[:, None] < batch_size) & (col_offsets[None, :] < out_features)
        x = tl.load(x_ptr + row_offsets[:, None] * out_features + col_offsets[None, :], mask=mask, other=-float('inf'))
        
        # Reduce along columns
        max_val = tl.maximum(max_val, tl.max(x, axis=1))
        sum_val += tl.sum(tl.where(mask, x, 0.0), axis=1)
    
    mean_val = sum_val / out_features
    
    # Compute final value: max - mean
    # The reference model computes max(x, dim=1) then subtracts mean(x, dim=1)
    # Note: The reference logic is: x = max(x, dim=1) - mean(x, dim=1)
    val = max_val - mean_val
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # sqrt(2/pi) approx 0.79788456
    cdf = 0.5 * (1.0 + tl.math.tanh(0.79788456 * (val + 0.044715 * val * val * val)))
    out = val * cdf
    
    # Store result
    store_mask = row_offsets < batch_size
    tl.store(out_ptr + row_offsets, out, mask=store_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.out_features = out_features

    def forward(self, x):
        x = self.gemm(x)
        batch_size = x.shape[0]
        out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_M = 16
        # Use a power of 2 for BLOCK_SIZE_N, but ensure it's at least 16
        BLOCK_SIZE_N = triton.next_power_of_2(min(self.out_features, 1024))
        
        grid = (triton.cdiv(batch_size, BLOCK_SIZE_M),)
        
        fused_post_gemm_kernel[grid](
            x, out,
            batch_size, self.out_features,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        return out