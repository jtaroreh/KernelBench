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
    
    # Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # Use tl.math.tanh and clamp exp to prevent overflow
    
    # First Mish
    # softplus(x) = ln(1 + exp(x))
    # For stability, use log1p(exp(x))
    softplus1 = tl.math.log(1.0 + tl.math.exp(tl.clamp(x, -88.0, 88.0)))
    mish1 = x * tl.math.tanh(softplus1)
    
    # Second Mish
    softplus2 = tl.math.log(1.0 + tl.math.exp(tl.clamp(mish1, -88.0, 88.0)))
    mish2 = mish1 * tl.math.tanh(softplus2)
    
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
        # Use standard linear layer
        x = self.linear(x)
        # Fuse the two Mish operations into a single kernel
        # Ensure input is contiguous for Triton
        return triton_mish_mish(x.contiguous())