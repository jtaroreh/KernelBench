import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_bias_groupnorm_kernel(
    x_ptr, bias_ptr, weight_ptr, mean_ptr, var_ptr, out_ptr,
    N, C, G, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # N: batch_size, C: out_features, G: num_groups
    # This kernel fuses Swish, Bias addition, and GroupNorm normalization
    # For simplicity in this implementation, we assume C is divisible by G
    
    pid = tl.program_id(0)
    group_size = C // G
    
    # Calculate offsets for the current batch row
    row_start = pid * C
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * C
    
    # 1. Swish + Bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + (offsets % C), mask=mask, other=0.0)
    swish = x * tl.sigmoid(x)
    val = swish + bias
    
    # 2. GroupNorm (Simplified: assumes mean/var pre-calculated or computed per group)
    # In a production scenario, one would use a two-pass approach or atomic reductions
    # Here we perform the normalization step assuming stats are provided
    group_idx = (offsets % C) // group_size
    mean = tl.load(mean_ptr + pid * G + group_idx)
    var = tl.load(var_ptr + pid * G + group_idx)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (val - mean) * inv_std
    
    # Apply weight/bias from GroupNorm if needed (omitted for brevity, usually identity)
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = self.matmul(x)
        
        # Compute stats for GroupNorm
        # Reshape to (N, G, C//G)
        N, C = x.shape
        G = self.group_norm.num_groups
        x_reshaped = x.view(N, G, -1)
        mean = x_reshaped.mean(dim=-1, keepdim=True)
        var = x_reshaped.var(dim=-1, keepdim=True, unbiased=False)
        
        # Launch fused kernel
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(C)
        
        swish_bias_groupnorm_kernel[(N,)](
            x, self.bias, self.group_norm.weight, 
            mean.squeeze(-1), var.squeeze(-1), out,
            N, C, G, self.group_norm.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Apply scale/shift from GroupNorm
        return out * self.group_norm.weight + self.group_norm.bias