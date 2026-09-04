import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_tanh_scale_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    spatial_size,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c_idx = (offsets // spatial_size) % C
    b_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Tanh identity: tanh(val) = 2.0 * sigmoid(2.0 * val) - 1.0
    t_val = 2.0 * tl.sigmoid(2.0 * val) - 1.0
    res = t_val * scaling_factor + b_val

    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model performing Conv2d followed by a fused Tanh, Scaling, and BiasAdd Triton kernel,
    and finally MaxPool2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        B, C, H, W = x.shape
        spatial_size = H * W
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_tanh_scale_bias_kernel[grid](
            x,
            self.bias,
            out,
            n_elements,
            C,
            spatial_size,
            float(self.scaling_factor),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        x = self.max_pool(out)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
