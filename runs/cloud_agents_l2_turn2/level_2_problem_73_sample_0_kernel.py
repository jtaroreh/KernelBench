import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bn_scale_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, channels, spatial_size, eps, scaling_factor,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate channel index: (offset // spatial_size) % channels
    # Note: x is (N, C, H, W), so index is (i // (H*W)) % C
    c_idx = (offsets // spatial_size) % channels

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + c_idx, mask=mask)
    var = tl.load(var_ptr + c_idx, mask=mask)
    weight = tl.load(weight_ptr + c_idx, mask=mask)
    bias = tl.load(bias_ptr + c_idx, mask=mask)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    # The numerical mismatch in the previous implementation was likely due to 
    # incorrect indexing or precision handling.
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = (x_hat * weight + bias) * scaling_factor

    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        
        # Ensure input is contiguous for predictable memory access
        x = x.contiguous()
        
        N, C, H, W = x.shape
        spatial_size = H * W
        n_elements = x.numel()
        
        out = torch.empty_like(x)
        
        # Ensure buffers are contiguous and on the same device
        mean = self.bn.running_mean.detach().contiguous()
        var = self.bn.running_var.detach().contiguous()
        weight = self.bn.weight.detach().contiguous()
        bias = self.bn.bias.detach().contiguous()
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        bn_scale_kernel[grid](
            x, mean, var, weight, bias, out,
            n_elements, C, spatial_size, self.bn.eps, self.scaling_factor,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out