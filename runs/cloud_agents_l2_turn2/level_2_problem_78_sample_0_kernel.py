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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for out-of-bounds
    mask = offsets < (batch * out_depth * out_height * out_width)
    
    # Calculate output coordinates
    b = (offsets // (out_depth * out_height * out_width))
    d = (offsets // (out_height * out_width)) % out_depth
    h = (offsets // out_width) % out_height
    w = offsets % out_width
    
    in_d_start = d * 6
    in_h_start = h * 6
    in_w_start = w * 6
    
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over channels
    for c in range(in_channels):
        # Initialize max_val with a very small number
        max_val = tl.full([BLOCK_SIZE], -1e20, dtype=tl.float32)
        
        # 6x6x6 window
        for id in range(6):
            for ih in range(6):
                for iw in range(6):
                    idx = (((b * in_channels + c) * depth + (in_d_start + id)) * height + (in_h_start + ih)) * width + (in_w_start + iw)
                    val = tl.load(x_ptr + idx, mask=mask, other=-1e20)
                    max_val = tl.maximum(max_val, val)
        
        sum_val += max_val
        
    tl.store(out_ptr + offsets, sum_val, mask=mask)

def triton_fused_pool_sum(x: torch.Tensor):
    batch, in_channels, depth, height, width = x.shape
    out_depth, out_height, out_width = depth // 6, height // 6, width // 6
    out = torch.empty((batch, 1, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)
    
    n_elements = out.numel()
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_pool_sum_kernel[grid](
        x, out,
        batch, in_channels, depth, height, width,
        out_depth, out_height, out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv_transpose(x)
        # The reference model performs MaxPool3d(2) then MaxPool3d(3), 
        # which is equivalent to a single MaxPool3d(6) window.
        return triton_fused_pool_sum(x)