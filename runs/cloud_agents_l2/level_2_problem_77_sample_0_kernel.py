import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_bn_pool_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    scale_factor, eps,
    N, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel fuses: x * scale -> BatchNorm -> GlobalAvgPool
    # Since GlobalAvgPool reduces spatial dims to 1x1x1, we compute the mean 
    # across D, H, W for each channel C.
    
    pid = tl.program_id(0)  # Channel index
    if pid >= C:
        return
    
    # Load BN parameters for this channel
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    weight = tl.load(weight_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Accumulate sum for global average pooling
    sum_val = 0.0
    spatial_size = D * H * W
    
    # Iterate over spatial dimensions
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Calculate flat index: batch * C * D * H * W + channel * D * H * W + spatial_idx
        # Assuming batch_size=16, we process one batch at a time or handle full tensor
        # For simplicity, this kernel assumes input is [N, C, D, H, W]
        # We compute global pool per channel across all batches and spatial dims
        # Or per batch per channel. Here we do per batch per channel.
        
        # Simplified: Global Avg Pool over D, H, W
        idx = pid * spatial_size + offsets
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        
        # Apply scale and BN
        x = (x * scale_factor - mean) * inv_std * weight + bias
        sum_val += tl.sum(x)
        
    tl.store(out_ptr + pid, sum_val / spatial_size)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Use Triton for fused scale + BN + GlobalAvgPool
        # BatchNorm3d stores running_mean/var. We use them for inference or training stats.
        N, C, D, H, W = x.shape
        out = torch.empty((N, C, 1, 1, 1), device=x.device)
        
        # For simplicity in this implementation, we call the fused kernel per batch
        for i in range(N):
            fused_scale_bn_pool_kernel[(C,)](
                x[i].data_ptr(), out[i].data_ptr(),
                self.batch_norm.running_mean.data_ptr(),
                self.batch_norm.running_var.data_ptr(),
                self.batch_norm.weight.data_ptr(),
                self.batch_norm.bias.data_ptr(),
                self.scale_factor, self.eps,
                N, C, D, H, W,
                BLOCK_SIZE=1024
            )
        return out