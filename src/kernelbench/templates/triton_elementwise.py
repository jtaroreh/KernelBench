import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def elementwise_unary_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Applied operation placeholder (e.g. relu, gelu, sigmoid)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_elementwise_unary(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    elementwise_unary_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
