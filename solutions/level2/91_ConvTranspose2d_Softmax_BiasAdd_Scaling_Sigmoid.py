import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_kernel(
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
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c_idx = (offsets // spatial_size) % C

    b_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    res = tl.sigmoid((val + b_val) * scaling_factor)

    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs ConvTranspose2d, softmax, and fuses
    bias addition, scaling, and sigmoid into a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = x.contiguous()
        out = torch.empty_like(x)

        B, C, H, W = x.shape
        n_elements = x.numel()
        spatial_size = H * W
        bias = self.bias.contiguous()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_bias_scale_sigmoid_kernel[grid](
            x,
            bias,
            out,
            n_elements,
            C,
            spatial_size,
            float(self.scaling_factor),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
