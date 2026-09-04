import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gn_leaky_relu_sum_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, num_channels, num_groups, eps, negative_slope,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    channels_per_group = num_channels // num_groups
    
    # GroupNorm requires statistics per group
    # We iterate over groups for this batch element
    for g in range(num_groups):
        group_start = g * channels_per_group
        
        # Compute mean and variance for the current group
        sum_val = 0.0
        sq_sum_val = 0.0
        
        for i in range(0, channels_per_group, BLOCK_SIZE_C):
            c_offsets = group_start + i + tl.arange(0, BLOCK_SIZE_C)
            mask = (c_offsets < (group_start + channels_per_group))
            x = tl.load(x_ptr + pid * num_channels + c_offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sq_sum_val += tl.sum(x * x, axis=0)
            
        mean = sum_val / channels_per_group
        var = (sq_sum_val / channels_per_group) - (mean * mean)
        inv_std = 1.0 / tl.sqrt(var + eps)
        
        # Apply normalization, scale, shift, leaky relu, and sum
        for i in range(0, channels_per_group, BLOCK_SIZE_C):
            c_offsets = group_start + i + tl.arange(0, BLOCK_SIZE_C)
            mask = (c_offsets < (group_start + channels_per_group))
            
            x = tl.load(x_ptr + pid * num_channels + c_offsets, mask=mask, other=0.0)
            w = tl.load(weight_ptr + c_offsets, mask=mask, other=1.0)
            b = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0)
            
            # GroupNorm
            norm = (x - mean) * inv_std
            out = norm * w + b
            
            # Leaky ReLU
            out = tl.where(out > 0, out, out * negative_slope)
            
            # Sum (x + x)
            out = out + out
            
            tl.store(out_ptr + pid * num_channels + c_offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        # Extract weights from the initialized GroupNorm layer
        self.weight = self.gn.weight
        self.bias = self.gn.bias

    def forward(self, x):
        x = self.fc(x)
        
        batch_size, hidden_size = x.shape
        out = torch.empty_like(x)
        
        # BLOCK_SIZE_C must be a power of 2 and divide channels_per_group
        channels_per_group = hidden_size // self.num_groups
        BLOCK_SIZE_C = 1
        while BLOCK_SIZE_C * 2 <= channels_per_group and BLOCK_SIZE_C < 1024:
            BLOCK_SIZE_C *= 2
        
        grid = (batch_size,)
        gn_leaky_relu_sum_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, hidden_size, self.num_groups, self.eps, self.negative_slope,
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        return out