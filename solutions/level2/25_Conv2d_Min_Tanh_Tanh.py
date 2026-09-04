import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh_impl(x):
    # Tanh(x) = 2 * sigmoid(2 * x) - 1
    # Sigmoid(x) = 1 / (1 + exp(-x))
    # Using identity: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # To avoid overflow, use: 2 * sigmoid(2 * x) - 1
    x2 = 2.0 * x
    # Clamp to avoid overflow in exp
    x2 = tl.clamp(x2, -20.0, 20.0)
    sigmoid = 1.0 / (1.0 + tl.exp(-x2))
    return 2.0 * sigmoid - 1.0

@triton.jit
def fused_min_tanh_tanh_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_spatial = batch_size * spatial_size
    
    if pid < num_spatial:
        b_idx = pid // spatial_size
        s_idx = pid % spatial_size
        
        min_val = 3.4028235e+38 # float32 max
        
        # Iterate over channels
        for c_start in range(0, out_channels, BLOCK_SIZE):
            c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
            mask = c_offsets < out_channels
            
            # Index: [batch, channel, spatial]
            # x is (batch, out_channels, height, width)
            offsets = (b_idx * out_channels * spatial_size) + (c_offsets * spatial_size) + s_idx
            vals = tl.load(x_ptr + offsets, mask=mask, other=3.4028235e+38)
            
            # Update running minimum
            min_val = tl.minimum(min_val, tl.min(vals, axis=0))
            
        # Apply double tanh
        res = tanh_impl(tanh_impl(min_val))
        
        # Store result
        out_offsets = (b_idx * spatial_size) + s_idx
        tl.store(out_ptr + out_offsets, res)

def triton_min_tanh_tanh(x: torch.Tensor):
    batch_size, out_channels, height, width = x.shape
    spatial_size = height * width
    out = torch.empty((batch_size, 1, height, width), device=x.device, dtype=x.dtype)
    
    # Use a block size that fits in SRAM
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
        # The kernel expects contiguous memory for efficient indexing
        return triton_min_tanh_tanh(x.contiguous())