import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_kernel(
    x_ptr,
    out_ptr,
    scale_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale_factor
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output using Triton kernel,
    applies batch normalization, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        fused_scale_kernel[grid](x, out, float(self.scale_factor), n_elements, BLOCK_SIZE=1024)
        x = self.batch_norm(out)
        x = self.global_avg_pool(x)
        return x


batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
