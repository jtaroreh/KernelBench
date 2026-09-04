import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gelu_pool_kernel(
    x_ptr, out_ptr,
    batch_size, out_channels, spatial_size,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    b = pid // out_channels
    c = pid % out_channels
    
    x_offset = (b * out_channels * spatial_size) + (c * spatial_size)
    
    sum_val = 0.0
    for i in range(0, spatial_size, BLOCK_SIZE_SPATIAL):
        offsets = i + tl.arange(0, BLOCK_SIZE_SPATIAL)
        mask = offsets < spatial_size
        
        # Load
        x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Use tl.math.tanh as tl.tanh is not available
        cdf = 0.5 * (1.0 + tl.math.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
        gelu_x = x * cdf
        
        sum_val += tl.sum(tl.where(mask, gelu_x, 0.0))
        
    tl.store(out_ptr + (b * out_channels + c), sum_val / spatial_size)

def triton_gelu_pool(x: torch.Tensor):
    batch_size, out_channels, h, w = x.shape
    spatial_size = h * w
    out = torch.empty((batch_size, out_channels), device=x.device, dtype=x.dtype)
    
    grid = (batch_size * out_channels,)
    # Ensure BLOCK_SIZE_SPATIAL is a power of 2
    BLOCK_SIZE_SPATIAL = 1
    while BLOCK_SIZE_SPATIAL < spatial_size and BLOCK_SIZE_SPATIAL < 1024:
        BLOCK_SIZE_SPATIAL *= 2
    
    gelu_pool_kernel[grid](
        x, out,
        batch_size, out_channels, spatial_size,
        BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fused GELU + Adaptive Avg Pool (1x1)
        return triton_gelu_pool(x)