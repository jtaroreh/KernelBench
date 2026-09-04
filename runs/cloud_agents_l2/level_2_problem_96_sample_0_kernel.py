import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_maxpool_global_avg_clamp_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    out_D, out_H, out_W,
    pool_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Fused kernel for:
    1. Scaling
    2. 3D MaxPool (with kernel_size=pool_size, stride=pool_size)
    3. Global Average Pooling (over the remaining pooled spatial dimensions)
    4. Clamp [0, 1]
    
    Each program handles one (batch, channel) pair: (b, c).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base pointers for this batch and channel
    bc_offset = b * (C * D * H * W) + c * (D * H * W)
    
    # We want to compute max pooling over blocks of size pool_size^3
    # out_D = D // pool_size, out_H = H // pool_size, out_W = W // pool_size
    # Then global average pool over out_D * out_H * out_W
    
    # Let's accumulate sum and count for the global average pool
    acc_sum = 0.0
    total_elements = out_D * out_H * out_W

    # Iterate over the pooled output grid
    # Since out_D, out_H, out_W are typically small (e.g. 8, 4, 2), we can loop inside the kernel safely.
    for od in range(out_D):
        for oh in range(out_H):
            for ow in range(out_W):
                # Find the max in the pooling window [od*pool_size : (od+1)*pool_size, ...]
                val_max = -float('inf')
                
                for pd in range(pool_size):
                    for ph in range(pool_size):
                        for pw in range(pool_size):
                            id_d = od * pool_size + pd
                            id_h = oh * pool_size + ph
                            id_w = ow * pool_size + pw
                            
                            idx = bc_offset + id_d * (H * W) + id_h * W + id_w
                            val = tl.load(x_ptr + idx)
                            val = val * scale
                            val_max = tl.maximum(val_max, val)
                
                acc_sum += val_max

    # Global average pooling over the pooled grid
    avg_val = acc_sum / total_elements

    # Clamp [0, 1]
    out_val = tl.maximum(0.0, tl.minimum(1.0, avg_val))

    # Store output shape: (B, C, 1, 1, 1)
    out_idx = b * C + c
    tl.store(out_ptr + out_idx, out_val)


def triton_fused_post_processing(x, scale, pool_size):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    
    B, C, D, H, W = x.shape
    out_D = D // pool_size
    out_H = H // pool_size
    out_W = W // pool_size
    
    out = torch.empty((B, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    grid = (B, C)
    
    scale_maxpool_global_avg_clamp_kernel[grid](
        x,
        out,
        B, C, D, H, W,
        out_D, out_H, out_W,
        pool_size=pool_size,
        scale=scale,
        BLOCK_SIZE_D=1,
        BLOCK_SIZE_HW=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_post_processing(x, self.scale, self.maxpool_kernel_size)
        return x