import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, channels, spatial_size,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    spatial_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_idx < spatial_size

    # Load all channels for these spatial locations to compute softmax
    # Softmax over dim=1 (channels)
    max_val = -float('inf')
    for c in range(channels):
        # Explicit pointer arithmetic: x_ptr is a pointer, c * spatial_size + spatial_idx is an int64 tensor
        ptr = x_ptr + (c * spatial_size + spatial_idx)
        val = tl.load(ptr, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, val)
    
    sum_exp = 0.0
    for c in range(channels):
        ptr = x_ptr + (c * spatial_size + spatial_idx)
        val = tl.load(ptr, mask=mask, other=-float('inf'))
        sum_exp += tl.exp(tl.clamp(val - max_val, -88.0, 88.0))
    
    # Apply operations
    for c in range(channels):
        ptr = x_ptr + (c * spatial_size + spatial_idx)
        val = tl.load(ptr, mask=mask, other=0.0)
        softmax_val = tl.exp(tl.clamp(val - max_val, -88.0, 88.0)) / sum_exp
        
        bias = tl.load(bias_ptr + c)
        
        # Add bias, scale, sigmoid
        res = (softmax_val + bias) * scaling_factor
        res_clamped = tl.clamp(res, -88.0, 88.0)
        out = 1.0 / (1.0 + tl.exp(-res_clamped))
        
        out_ptr_c = out_ptr + (c * spatial_size + spatial_idx)
        tl.store(out_ptr_c, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Prepare for Triton
        batch, c, h, w = x.shape
        x_flat = x.view(batch * c, h * w).contiguous()
        out = torch.empty_like(x_flat)
        
        spatial_size = h * w
        BLOCK_SIZE = 64
        grid = (triton.cdiv(spatial_size, BLOCK_SIZE),)
        
        # We iterate over batches to keep kernel logic simple
        for b in range(batch):
            x_ptr = x_flat.data_ptr() + b * c * spatial_size * 4
            out_ptr = out.data_ptr() + b * c * spatial_size * 4
            fused_post_conv_kernel[grid](
                x_ptr,
                self.bias.data_ptr(),
                out_ptr,
                c * spatial_size, c, spatial_size,
                self.scaling_factor,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
        return out.view(batch, c, h, w)