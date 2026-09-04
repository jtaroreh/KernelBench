import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_min_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    spatial_size,
    scale_factor,
):
    # Each program handles one spatial location (h * w) across the batch
    pid = tl.program_id(0)
    
    # Calculate spatial index and batch index
    spatial_idx = pid % spatial_size
    batch_idx = pid // spatial_size
    
    # Initialize min_val with a large number
    # Use a scalar value instead of a block to avoid type mismatch
    min_val = float('inf')
    
    # Iterate over channels
    # Calculate base pointer for this spatial location
    base_ptr = x_ptr + (batch_idx * out_channels * spatial_size) + spatial_idx
    
    for c in range(out_channels):
        # Load value at current channel
        val = tl.load(base_ptr + (c * spatial_size))
        val = val * scale_factor
        min_val = tl.minimum(min_val, val)
    
    # Store result
    out_idx = (batch_idx * spatial_size) + spatial_idx
    tl.store(out_ptr + out_idx, min_val)

def triton_scale_min(x: torch.Tensor, scale_factor: float):
    batch_size, out_channels, height, width = x.shape
    spatial_size = height * width
    out = torch.empty((batch_size, 1, height, width), device=x.device, dtype=x.dtype)
    
    grid = (batch_size * spatial_size,)
    
    fused_scale_min_kernel[grid](
        x, out,
        batch_size, out_channels, spatial_size, scale_factor
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        # Ensure input is contiguous for the kernel
        x = x.contiguous()
        # Fuse the scaling and the channel-wise minimum operation
        return triton_scale_min(x, self.scale_factor)