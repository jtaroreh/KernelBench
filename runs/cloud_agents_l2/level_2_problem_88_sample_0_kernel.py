import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_norm_swish_mul_swish_kernel(
    x_ptr, weight_ptr, out_ptr,
    n_elements, n_features, num_groups,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel fuses: GroupNorm + Swish + Multiply + Swish
    # Note: For simplicity and performance, we assume GroupNorm statistics 
    # are pre-calculated or handled via a simplified normalization approach.
    # Here we implement the activation/multiplication chain.
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(weight_ptr + (offsets % n_features), mask=mask)
    
    # Swish 1: x * sigmoid(x)
    s1 = x * tl.sigmoid(x)
    # Multiply
    m = s1 * w
    # Swish 2: m * sigmoid(m)
    out = m * tl.sigmoid(m)
    
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.out_features = out_features

    def forward(self, x):
        # GEMM remains standard (cuBLAS is highly optimized)
        x = self.gemm(x)
        
        # GroupNorm
        x = self.group_norm(x)
        
        # Fused Swish + Multiply + Swish
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fused_norm_swish_mul_swish_kernel[grid](
            x, self.multiply_weight, out,
            n_elements, self.out_features, 256,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out