import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_hardswish_norm_mean_kernel(
    x_ptr, out_ptr,
    batch, channels, spatial_size,
    num_groups, eps,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fuses HardSwish, GroupNorm, and Spatial Mean.
    GroupNorm: (B, C, S) -> (B, G, C/G, S)
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    
    # Calculate group statistics
    c_per_g = channels // num_groups
    group_start_c = pid_g * c_per_g
    
    # Accumulate sum and sum_sq for variance
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # Load data and compute HardSwish + stats
    # HardSwish(x) = x * min(max(x + 3, 0), 6) / 6
    for c_off in range(c_per_g):
        c = group_start_c + c_off
        for s in range(spatial_size):
            idx = (pid_b * channels * spatial_size) + (c * spatial_size) + s
            val = tl.load(x_ptr + idx)
            
            # HardSwish
            hswish = val * tl.clamp(val + 3.0, 0.0, 6.0) / 6.0
            
            sum_val += hswish
            sum_sq_val += hswish * hswish
            
    # Compute mean and inv_std
    mean = sum_val / (c_per_g * spatial_size)
    var = (sum_sq_val / (c_per_g * spatial_size)) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and compute final spatial mean per channel
    for c_off in range(c_per_g):
        c = group_start_c + c_off
        channel_sum = 0.0
        for s in range(spatial_size):
            idx = (pid_b * channels * spatial_size) + (c * spatial_size) + s
            val = tl.load(x_ptr + idx)
            hswish = val * tl.clamp(val + 3.0, 0.0, 6.0) / 6.0
            norm_val = (hswish - mean) * inv_std
            channel_sum += norm_val
            
        out_idx = (pid_b * channels) + c
        tl.store(out_ptr + out_idx, channel_sum / spatial_size)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        spatial_size = D * H * W
        
        # Flatten spatial dims for kernel
        x_flat = x.view(B, C, spatial_size)
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)
        
        grid = (B, self.num_groups)
        fused_hardswish_norm_mean_kernel[grid](
            x_flat, out,
            B, C, spatial_size,
            self.num_groups, self.eps,
            BLOCK_SIZE_C=C // self.num_groups
        )
        return out