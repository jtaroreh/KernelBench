import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(x)
    swish = x * tl.sigmoid(x)
    
    # Divide by 2.0
    val = swish / 2.0
    
    # Clamp [-1, 1]
    val = tl.maximum(tl.minimum(val, 1.0), -1.0)
    
    # Tanh
    val = tl.tanh(val)
    
    # Clamp [-1, 1]
    val = tl.maximum(tl.minimum(val, 1.0), -1.0)
    
    # Store
    tl.store(out_ptr + offsets, val, mask=mask)

def triton_fused_activation(x: torch.Tensor):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_activation_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Perform GEMM using optimized PyTorch implementation
        x = self.gemm(x)
        # Perform all subsequent element-wise operations in a single fused Triton kernel
        return triton_fused_activation(x)