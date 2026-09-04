import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mish_subtraction_kernel(
    x_ptr,
    out_ptr,
    sub1,
    sub2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Subtract
    x = x - sub1 - sub2
    
    # Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # To avoid overflow in exp(x), clamp x. 
    # For x > 20, softplus(x) is approximately x.
    # For x < -20, softplus(x) is approximately exp(x).
    
    # Clamp x to prevent overflow in tl.exp
    x_clamped = tl.clamp(x, -20.0, 20.0)
    softplus = tl.log(1.0 + tl.exp(x_clamped))
    
    # For large x, softplus(x) -> x. For small x, softplus(x) -> log(1+exp(x))
    # We use tl.where to handle the numerical stability of softplus
    softplus = tl.where(x > 20.0, x, softplus)
    
    out = x * tl.math.tanh(softplus)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_mish_subtraction(x: torch.Tensor, sub1: float, sub2: float):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    mish_subtraction_kernel[grid](
        x, out, sub1, sub2, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.sub1 = float(subtract_value_1)
        self.sub2 = float(subtract_value_2)

    def forward(self, x):
        x = self.conv(x)
        # Fusing the two subtractions and the Mish activation into one kernel
        return triton_mish_subtraction(x, self.sub1, self.sub2)