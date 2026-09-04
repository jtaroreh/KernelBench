import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def fused_mish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sp = tl.where(val > 20.0, val, tl.log(1.0 + tl.exp(tl.clamp(val, -88.0, 88.0))))
    m = val * _tanh(sp)
    tl.store(out_ptr + offsets, m, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies fused Mish activation
    using a custom Triton kernel, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x = self.bn(out)
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
