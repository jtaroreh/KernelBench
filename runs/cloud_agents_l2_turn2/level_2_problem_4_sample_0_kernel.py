import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def double_mish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the block start offset for the current program
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Mish(x) = x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # To avoid overflow in exp(x), we use the identity:
    # softplus(x) = x + log(1 + exp(-x)) if x > 0 else log(1 + exp(x))
    # Or simply clamp the input to exp to prevent overflow.
    
    def mish(val):
        # Clamp to prevent exp overflow
        exp_val = tl.exp(tl.clamp(val, -88.0, 88.0))
        softplus = tl.log(1.0 + exp_val)
        return val * tl.math.tanh(softplus)

    # First Mish
    y = mish(x)
    # Second Mish
    z = mish(y)

    # Store the final result
    tl.store(out_ptr + offsets, z, mask=mask)

def triton_double_mish(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    double_mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that replaces two consecutive Mish activations with a
    single, fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_double_mish(x)
        return x