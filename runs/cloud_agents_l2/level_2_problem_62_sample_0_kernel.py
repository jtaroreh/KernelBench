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
    # Each program handles one batch element
    pid = tl.program_id(0)
    
    channels_per_group = num_channels // num_groups
    
    # Calculate mean and variance for GroupNorm
    mean = 0.0
    var = 0.0
    
    # Load data and compute statistics
    for i in range(0, num_channels, BLOCK_SIZE_C):
        offsets = pid * num_channels + i + tl.arange(0, BLOCK_SIZE_C)
        mask = (i + tl.arange(0, BLOCK_SIZE_C)) < num_channels
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
        var += tl.sum(x * x, axis=0)
    
    mean /= num_channels
    var = (var / num_channels) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization, scale, shift, leaky relu, and sum
    for i in range(0, num_channels, BLOCK_SIZE_C):
        c_offsets = i + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < num_channels
        
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
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = self.fc(x)
        
        batch_size, hidden_size = x.shape
        out = torch.empty_like(x)
        
        # Triton kernel launch
        # BLOCK_SIZE_C should be a power of 2
        BLOCK_SIZE_C = triton.next_power_of_2(hidden_size // self.num_groups)
        
        grid = (batch_size,)
        gn_leaky_relu_sum_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, hidden_size, self.num_groups, self.eps, self.negative_slope,
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        return out