import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    spatial_size,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    channel_idx = (offsets // spatial_size) % num_channels
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    out = tl.maximum(x + bias_val, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_fused_bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    num_channels = x.shape[1] if x.ndim >= 2 else 1
    spatial_size = 1
    for s in x.shape[2:]:
        spatial_size *= s
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_bias_relu_kernel[grid](x, bias, out, n_elements, spatial_size, num_channels, BLOCK_SIZE=BLOCK_SIZE)
    return out
