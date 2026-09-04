import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_norm_add_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements, n_features, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel performs InstanceNorm (per-row), Add, and Mul in one pass
    # to maximize memory bandwidth utilization.
    pid = tl.program_id(0)
    row_start = pid * n_features
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_start + n_features)

    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Instance Norm (Mean and Var calculation for the row)
    mean = tl.sum(x, axis=0) / n_features
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_features
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    x_norm = (x - mean) * inv_std
    
    # Residual Add and Mul
    out = (x_norm + y) * y
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_fused_norm_add_mul(x: torch.Tensor, y: torch.Tensor, eps: float):
    batch_size, n_features = x.shape
    out = torch.empty_like(x)
    
    # BLOCK_SIZE must be a power of 2 >= n_features
    BLOCK_SIZE = triton.next_power_of_2(n_features)
    
    grid = (batch_size,)
    fused_norm_add_mul_kernel[grid](
        x, y, out,
        batch_size * n_features, n_features, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, y):
        # Linear layer remains standard, but we fuse the subsequent operations
        x = self.linear(x)
        # Replace InstanceNorm2d + Add + Mul with fused Triton kernel
        return triton_fused_norm_add_mul(x, y, self.eps)