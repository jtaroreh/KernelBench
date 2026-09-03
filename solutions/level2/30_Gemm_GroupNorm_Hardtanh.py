import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_hardtanh_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, G,
    min_val, max_val,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
):
    # N: batch_size, C: out_features, G: num_groups
    # Each program handles one row (batch element)
    pid = tl.program_id(0)
    
    # Group normalization parameters
    C_per_G = C // G
    
    # Calculate mean and variance for the group
    # We iterate over groups for the current batch element
    for g in range(G):
        # Load group data
        group_offset = g * C_per_G
        cols = tl.arange(0, BLOCK_SIZE_C)
        mask = cols < C_per_G
        
        x_vals = tl.load(x_ptr + pid * C + group_offset + cols, mask=mask, other=0.0)
        
        # Compute mean
        mean = tl.sum(x_vals, axis=0) / C_per_G
        
        # Compute variance
        var = tl.sum((x_vals - mean) * (x_vals - mean), axis=0) / C_per_G
        inv_std = 1.0 / tl.sqrt(var + eps)
        
        # Normalize and apply affine + HardTanh
        w = tl.load(weight_ptr + group_offset + cols, mask=mask, other=1.0)
        b = tl.load(bias_ptr + group_offset + cols, mask=mask, other=0.0)
        
        out = (x_vals - mean) * inv_std * w + b
        out = tl.maximum(min_val, tl.minimum(max_val, out))
        
        tl.store(out_ptr + pid * C + group_offset + cols, out, mask=mask)

def triton_group_norm_hardtanh(x, weight, bias, num_groups, min_val, max_val):
    N, C = x.shape
    out = torch.empty_like(x)
    eps = 1e-5
    
    # BLOCK_SIZE_C should be a power of 2 >= C/G
    BLOCK_SIZE_C = triton.next_power_of_2(C // num_groups)
    
    grid = (N,)
    group_norm_hardtanh_kernel[grid](
        x, weight, bias, out,
        N, C, num_groups,
        min_val, max_val,
        eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Register GN parameters as buffers
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = self.gemm(x)
        # Fused GroupNorm + HardTanh
        return triton_group_norm_hardtanh(
            x, self.weight, self.bias, 
            self.num_groups, self.hardtanh_min, self.hardtanh_max
        )