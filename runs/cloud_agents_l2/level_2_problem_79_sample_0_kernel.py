import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, m_ptr, out_ptr,
    n_elements, channels, spatial_size,
    clamp_min, clamp_max,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate channel index for multiplier
    # x is (N, C, D, H, W), spatial_size = D*H*W
    idx = offsets // spatial_size
    c_idx = (idx % (channels * spatial_size)) // spatial_size
    
    x = tl.load(x_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + c_idx, mask=mask)
    
    # Multiply -> Clamp -> Multiply
    x = x * m
    # Instance Norm is complex to fuse fully due to mean/var, 
    # so we handle the element-wise ops here
    x = tl.maximum(tl.minimum(x, clamp_max), clamp_min)
    x = x * m
    
    tl.store(out_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        
        # Prepare for Triton
        N, C, D, H, W = x.shape
        spatial_size = D * H * W
        n_elements = x.numel()
        
        out = torch.empty_like(x)
        
        # Launch fused kernel for: x * m -> clamp -> x * m
        # Note: InstanceNorm is performed separately as it requires reduction
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_post_conv_kernel[grid](
            x, self.multiplier.view(-1), out,
            n_elements, C, spatial_size,
            self.clamp_min, self.clamp_max,
            BLOCK_SIZE=1024
        )
        
        # Apply InstanceNorm
        x = self.instance_norm(out)
        
        # Max over channel dim
        x = torch.max(x, dim=1)[0]
        return x