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
    """
    Fused kernel for Swish, Divide, Clamp, Tanh, and Clamp operations.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(x)
    # tl.sigmoid is available in triton.language
    sig = tl.sigmoid(x)
    x = x * sig
    
    # Divide by 2.0
    x = x / 2.0
    
    # Clamp between -1.0 and 1.0
    x = tl.maximum(tl.minimum(x, 1.0), -1.0)
    
    # Tanh: Use tl.math.tanh as tl.tanh is not available in triton.language
    x = tl.math.tanh(x)
    
    # Clamp between -1.0 and 1.0
    x = tl.maximum(tl.