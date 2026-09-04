import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, sub_ptr, out_ptr,
    batch, channels, depth, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_spatial = batch * depth * height * width
    
    spatial_idx = pid
    if spatial_idx < num_spatial:
        # 1. Softmax across channels
        # Load all channels for this spatial location
        # Since __setitem__ is not supported, we load and compute max/exp in a pass
        
        # First pass: find max for numerical stability
        max_val = -1000.0
        for c in range(channels):
            idx = spatial_idx * channels + c
            val = tl.load(x_ptr + idx)
            if val > max_val:
                max_val = val
        
        # Second pass: compute sum of exponentials
        sum_exp = 0.0
        for c in range(channels):
            idx = spatial_idx * channels + c
            val = tl.load(x_ptr + idx)
            sum_exp += tl.exp(tl.clamp(val - max_val, -88.0, 88.0))
        
        # Third pass: compute softmax, subtract, swish, and find max across channels
        max_swish = -1000.0
        for c in range(channels):
            idx = spatial_idx * channels + c
            val = tl.load(x_ptr + idx)
            s = tl.exp(tl.clamp(val - max_val, -88.0, 88.0)) / sum_exp
            
            sub = tl.load(sub_ptr + c)
            val_sub = s - sub
            
            # Swish: x * sigmoid(x)
            sigmoid_val = 1.0 / (1.0 + tl.exp(tl.clamp(-val_sub, -88.0, 88.0)))
            swish = val_sub * sigmoid_val
            
            if swish > max_swish:
                max_swish = swish
        
        tl.store(out_ptr + spatial_idx, max_swish)

class ModelNew(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        
        B, C, D, H, W = x.shape
        out = torch.empty((B, D, H, W), device=x.device, dtype=x.dtype)
        
        # Flatten spatial dims for kernel: (B, C, D, H, W) -> (B * D * H * W, C)
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        n_spatial = B * D * H * W
        grid = lambda meta: (n_spatial,)
        
        fused_post_conv_kernel[grid](
            x_flat, self.subtract, out,
            B, C, D, H, W,
            BLOCK_SIZE=C
        )
        
        return out