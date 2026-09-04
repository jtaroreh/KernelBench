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
    
    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # Using tl.exp and tl.log for stability
    # softplus(x) = log(1 + exp(x))
    # tanh(softplus(x)) = (exp(softplus(x)) - exp(-softplus(x))) / (exp(softplus(x)) + exp(-softplus(x)))
    # exp(softplus(x)) = 1 + exp(x)
    # exp(-softplus(x)) = 1 / (1 + exp(x))
    
    exp_x = tl.exp(x)
    softplus = tl.log(1.0 + exp_x)
    out = x * tl.tanh(softplus)
    
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