import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_gelu_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols, eps, scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (spatial location)
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * n_cols
    
    # Load row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(row_ptr + cols, mask=mask, other=0.0)
    
    # LayerNorm: Mean and Variance
    # Use mask for sum to handle cases where BLOCK_SIZE > n_cols
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0