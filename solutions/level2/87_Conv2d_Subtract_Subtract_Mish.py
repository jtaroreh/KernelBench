import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _mish(x):
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.clamp(x, -88.0, 88.0))))
    return x * _tanh(sp)


@triton.jit
def fused_sub_mish_kernel(x_ptr, out_ptr, sub_val, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    subbed = val - sub_val
    res = _mish(subbed)
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution and fuses subtraction
    of two values and Mish activation into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        x = self.conv(x)
        out = torch.empty_like(x)
        sub_total = float(self.subtract_value_1 + self.subtract_value_2)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_sub_mish_kernel[grid](x, out, sub_total, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
