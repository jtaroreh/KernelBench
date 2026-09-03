import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_min_tanh_tanh_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location (h * w) across the batch
    pid = tl.program_id(0)
    num_spatial = batch_size * spatial_size
    
    if pid < num_spatial:
        # Calculate batch and spatial index
        b_idx = pid // spatial_size
        s_idx = pid % spatial_size
        
        # Find min across channels
        min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
        
        # Iterate over channels in blocks
        for c_start in range(0, out_channels, BLOCK_SIZE):
            c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
            mask = c_offsets < out_channels
            
            # Index: [batch, channel, spatial]
            offsets = (b_idx * out_channels * spatial_size) + (c_offsets * spatial_size) + s_idx
            vals = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
            min_val = tl.minimum(min_val, vals)
            
        # Reduce min_val across the block
        final_min = tl.min(min_val, axis=0)
        
        # Apply double tanh
        res = tl.tanh(tl.tanh(final_min))
        
        # Store result
        out_offsets = (b_idx * spatial_size) + s_idx
        tl.store(out_ptr + out_offsets, res)

def triton_min_tanh_tanh(x: torch.Tensor):
    batch_size, out_channels, height, width = x.shape
    spatial_size = height * width
    out = torch.empty((batch_size, 1, height, width), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(out_channels)
    grid = (batch_size * spatial_size,)
    
    fused_min_tanh_tanh_kernel[grid](
        x, out, batch_size, out_channels, spatial_size, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fusing min, tanh, and tanh into a single kernel
        return triton_min_tanh_tanh(x)