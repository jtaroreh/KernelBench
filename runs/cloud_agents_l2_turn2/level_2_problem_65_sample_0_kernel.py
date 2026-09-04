import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pool_sigmoid_sum_kernel(
    x_ptr, out_ptr,
    batch, channels, height, width,
    pool_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Grid: (batch * channels)
    pid = tl.program_id(0)
    b = pid // channels
    c = pid % channels
    
    # Output spatial dimensions after pooling
    out_h = height // pool_size
    out_w = width // pool_size
    
    sum_val = 0.0
    
    # Iterate over output spatial grid
    for oh in range(out_h):
        for ow in range(out_w):
            # Average pooling: sum over pool_size x pool_size window
            local_sum = 0.0
            for ph in range(pool_size):
                for pw in range(pool_size):
                    ih = oh * pool_size + ph
                    iw = ow * pool_size + pw
                    idx = ((b * channels + c) * height + ih) * width + iw
                    local_sum += tl.load(x_ptr + idx)
            
            avg = local_sum / (pool_size * pool_size)
            # Sigmoid
            sig = 1.0 / (1.0 + tl.exp(-avg))
            sum_val += sig
            
    tl.store(out_ptr + pid, sum_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        
        batch, channels, height, width = x.shape
        out = torch.empty((batch, channels), device=x.device, dtype=x.dtype)
        
        # Launch kernel to fuse pooling, sigmoid, and spatial sum
        grid = (batch * channels,)
        pool_sigmoid_sum_kernel[grid](
            x, out,
            batch, channels, height, width,
            pool_size=self.pool_size,
            BLOCK_C=1
        )
        
        # Sum over channels
        return out.sum(dim=1)