import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_scale_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    swish_x = x * sigmoid_x
    
    # Scale
    out = swish_x * scaling_factor
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_swish_scale(x: torch.Tensor, scaling_factor: float):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    swish_scale_kernel[grid](x, out, n_elements, scaling_factor, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor) -> None:
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Use standard cuBLAS for matmul (highly optimized)
        x = self.matmul(x)
        # Use fused Triton kernel for Swish + Scaling
        return triton_swish_scale(x, self.scaling_factor)