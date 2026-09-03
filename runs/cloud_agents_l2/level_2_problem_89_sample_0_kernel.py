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
    # This kernel performs: Softmax(dim=1) -> Subtract -> Swish -> Max(dim=1)
    # Since Softmax requires a reduction across channels, we perform it in two passes
    # or use a simplified approach if channels are small. Here we assume channels=16.
    
    pid = tl.program_id(0)
    num_spatial = depth * height * width
    
    # Calculate spatial index
    spatial_idx = pid
    if spatial_idx < num_spatial:
        # 1. Softmax across channels
        # Load all channels for this spatial location
        acc = tl.zeros([16], dtype=tl.float32)
        for c in range(channels):
            idx = (spatial_idx * channels + c)
            val = tl.load(x_ptr + idx)
            acc[c] = tl.exp(val)
        
        sum_exp = tl.sum(acc, axis=0)
        
        # 2. Subtract, Swish, and Max
        max_val = -1e20
        for c in range(channels):
            # Softmax
            s = acc[c] / sum_exp
            # Subtract
            sub = tl.load(sub_ptr + c)
            val = s - sub
            # Swish
            swish = val * (1.0 / (1.0 + tl.exp(-val)))
            # Max reduction
            if swish > max_val:
                max_val = swish
        
        tl.store(out_ptr + spatial_idx, max_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        
        # Prepare output
        B, C, D, H, W = x.shape
        out = torch.empty((B, D, H, W), device=x.device)
        
        # Flatten spatial dims for kernel
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        
        # Launch kernel
        # We process each spatial location (B*D*H*W)
        n_spatial = B * D * H * W
        def grid(meta): return (n_spatial,)
        
        fused_post_conv_kernel[grid](
            x_flat, self.subtract, out,
            B, C, D, H, W,
            BLOCK_SIZE=16
        )
        
        return out