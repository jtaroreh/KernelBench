import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mish_tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform fused Mish and Tanh activations.
    Mish(x) = x * tanh(softplus(x))
    Softplus(x) = log(1 + exp(x))
    Final output = tanh(Mish(x))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Softplus: log(1 + exp(x))
    # For numerical stability, we use the identity log(1 + exp(x)) ≈ x for large x.
    # We also clamp the input to exp to avoid overflow.
    exp_x = tl.math.exp(tl.clamp(x, -88.0, 88.0))
    softplus = tl.where(x > 20.0, x, tl.math.log(1.0 + exp_x))
    
    # Mish: x * tanh