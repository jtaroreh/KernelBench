import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scaling_factor
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies Batch Normalization, and scales the output using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_scale_kernel[grid](
            x,
            out,
            n_elements,
            float(self.scaling_factor),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
