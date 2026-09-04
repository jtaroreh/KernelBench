import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_pool_sum_kernel(
    x_ptr, out_ptr,
    batch, in_channels, depth, height, width,
    out_depth, out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    # This kernel performs the two MaxPool3d operations and the Sum reduction
    # in a single pass to maximize cache locality.
    # Pool1: 2x2x2, Pool2: 3x3x3 -> Total reduction factor 6x6x6
    
    pid = tl.program_id(0)
    num_elements = batch * out_depth * out_height * out_width
    
    if pid * BLOCK_SIZE >= num_elements:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate output coordinates
    b = (offsets // (out_depth * out_height * out_width))
    d = (offsets // (out_height * out_width)) % out_depth
    h = (offsets // out_width) % out_height
    w = offsets % out_width
    
    # Input coordinates for the 6x6x6 window
    in_d_start = d * 6
    in_h_start = h * 6
    in_w_start = w * 6
    
    sum_val = 0.0
    for c in range(in_channels):
        max_val = -1e20
        for id in range(6):
            for ih in range(6):
                for iw in range(6):
                    idx = (((b * in_channels + c) * depth + (in_d_start + id)) * height + (in_h_start + ih)) * width + (in_w_start + iw)
                    val = tl.load(x_ptr + idx)
                    max_val = tl.maximum(max_val, val)
        sum_val += max_val
        
    tl.store(out_ptr + offsets, sum_val)

def triton_fused_pool_sum(x: torch.Tensor):
    batch, in_channels, depth, height, width = x.shape
    # After 2x2x2 and 3x3x3 pooling, dimensions are reduced by 6
    out_depth, out_height, out_width = depth // 6, height // 6, width // 6
    out = torch.empty((batch, 1, out_depth, out_height, out_width), device=x.device)
    
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_pool_sum_kernel[grid](
        x, out,
        batch, in_channels, depth, height, width,
        out_depth, out_height, out_width,
        BLOCK_SIZE=32
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Replace MaxPool3d(2), MaxPool3d(3), and sum(dim=1) with fused kernel
        return triton_fused_pool_sum(x)