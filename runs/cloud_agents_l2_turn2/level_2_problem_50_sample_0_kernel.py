import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, bias_ptr, out_ptr,
    scale1, scale2,
    n_elements,
    channels, spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply scale1
    x = x * scale1
    
    # Apply bias (broadcasted over spatial dimensions)
    # Index in channel dimension: (offsets // spatial_size) % channels
    c_idx = (offsets // spatial_size) % channels
    bias = tl.load(bias_ptr + c_idx, mask=c_idx < channels, other=0.0)
    
    x = x + bias
    
    # Apply scale2
    x = x * scale2
    
    tl.store(out_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = scale1
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze())
        self.scale2 = scale2
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        
        # Triton fusion for: x * scale1 + bias, then * scale2
        # Note: AvgPool3d changes spatial dimensions, so we apply fusion after pooling
        n_elements = x.numel()
        spatial_size = x.shape[2] * x.shape[3] * x.shape[4]
        out = torch.empty_like(x)
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        
        fused_post_conv_kernel[grid](
            x, self.bias, out,
            self.scale1, self.scale2,
            n_elements, self.out_channels, spatial_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out