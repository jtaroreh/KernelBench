import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, m_ptr, out_ptr,
    n_elements, channels, spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x and multiplier
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Calculate channel index to broadcast multiplier
    # x is (N, C, H, W), multiplier is (C, 1, 1)
    channel_idx = (offsets // spatial_size) % channels
    m = tl.load(m_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
    
    # Apply multiplier
    x = x * m
    
    # LeakyReLU (alpha=0.01)
    x = tl.where(x > 0, x, x * 0.01)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    
    tl.store(out_ptr + offsets, gelu, mask=mask)

def triton_fused_post_conv(x: torch.Tensor, multiplier: torch.Tensor):
    n_elements = x.numel()
    channels = x.shape[1]
    spatial_size = x.shape[2] * x.shape[3]
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_post_conv_kernel[grid](
        x, multiplier, out,
        n_elements, channels, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        x = self.conv(x)
        # Fuse multiplication, LeakyReLU, and GELU into one kernel
        return triton_fused_post_conv(x, self.multiplier.view(-1))