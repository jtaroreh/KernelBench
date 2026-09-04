import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_bn_pool_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    scale_factor, eps,
    C, spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load BN parameters for this channel
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    weight = tl.load(weight_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    sum_val = 0.0
    
    # Iterate over spatial dimensions
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Calculate pointer: x_ptr is offset by (pid * spatial_size)
        # We use tl.make_block_ptr or simple pointer arithmetic
        # Ensure pointers are float32 (triton requires explicit pointer types)
        x_offsets = pid * spatial_size + offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
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
        N, C, D, H, W = x.shape
        spatial_size = D * H * W
        out = torch.empty((N, C, 1, 1, 1), device=x.device)
        
        # Ensure BN parameters are in float32 for the kernel
        mean = self.batch_norm.running_mean.float()
        var = self.batch_norm.running_var.float()
        weight = self.batch_norm.weight.float()
        bias = self.batch_norm.bias.float()
        
        for i in range(N):
            fused_scale_bn_pool_kernel[(C,)](
                x[i].data_ptr(), out[i].data_ptr(),
                mean.data_ptr(),
                var.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                self.scale_factor, self.eps,
                C, spatial_size,
                BLOCK_SIZE=1024
            )
        return out