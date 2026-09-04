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
    # This kernel performs: Softmax(dim=1) -> Add Bias -> Scale -> Sigmoid
    # We process spatial locations in parallel.
    pid = tl.program_id(0)
    spatial_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_idx < spatial_size

    # Load all channels for these spatial locations to compute softmax
    # Softmax over dim=1 (channels)
    max_val = -float('inf')
    for c in range(channels):
        val = tl.load(x_ptr + (c * spatial_size + spatial_idx), mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, val)
    
    sum_exp = 0.0
    for c in range(channels):
        val = tl.load(x_ptr + (c * spatial_size + spatial_idx), mask=mask, other=-float('inf'))
        sum_exp += tl.exp(val - max_val)
    
    # Apply operations
    for c in range(channels):
        val = tl.load(x_ptr + (c * spatial_size + spatial_idx), mask=mask, other=0.0)
        softmax_val = tl.exp(val - max_val) / sum_exp
        
        bias = tl.load(bias_ptr + c)
        
        # Add bias, scale, sigmoid
        res = (softmax_val + bias) * scaling_factor
        out = 1.0 / (1.0 + tl.exp(-res))
        
        tl.store(out_ptr + (c * spatial_size + spatial_idx), out, mask=mask)

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
        x = x.view(batch * c, h * w)
        out = torch.empty_like(x)
        
        spatial_size = h * w
        BLOCK_SIZE = 64
        grid = (triton.cdiv(spatial_size, BLOCK_SIZE),)
        
        # We iterate over batches to keep kernel logic simple
        for b in range(batch):
            fused_post_conv_kernel[grid](
                x.data_ptr() + b * c * spatial_size * 4,
                self.bias.data_ptr(),
                out.data_ptr() + b * c * spatial_size * 4,
                c * spatial_size, c, spatial_size,
                self.scaling_factor,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
        return out.view(batch, c, h, w)