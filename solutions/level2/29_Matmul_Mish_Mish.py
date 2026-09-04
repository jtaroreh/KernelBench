import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mish_mish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Mish(x) = x * tanh(softplus(x))
    # Tanh(x) = 2 * sigmoid(2 * x) - 1
    # Softplus(x) = ln(1 + exp(x))
    
    # First Mish
    # Use tl.exp with clamping for stability
    softplus1 = tl.log(1.0 + tl.exp(tl.clamp(x, -88.0, 88.0)))
    # tanh(softplus1) = 2 * sigmoid(2 * softplus1) - 1
    tanh1 = 2.0 * tl.sigmoid(2.0 * softplus1) - 1.0
    mish1 = x * tanh1
    
    # Second Mish
    softplus2 = tl.log(1.0 + tl.exp(tl.clamp(mish1, -88.0, 88.0)))
    tanh2 = 2.0 * tl.sigmoid(2.0 * softplus2) - 1.0
    mish2 = mish1 * tanh2
    
    tl.store(out_ptr + offsets, mish2, mask=mask)

def triton_mish_mish(x: torch.Tensor):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mish_mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Perform linear transformation
        x = self.linear(x)
        # Apply fused Mish-Mish kernel
        # Ensure input is contiguous for Triton memory access
        return triton_mish_mish(x.contiguous())