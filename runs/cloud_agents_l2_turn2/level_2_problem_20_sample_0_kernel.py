import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    y_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    S,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c_idx = (offsets // S) % C

    y = tl.load(y_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + c_idx, mask=mask)

    x = y + b
    x = x + y
    x = x * y
    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_ops(y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = y.contiguous()
    bias = bias.contiguous()

    N, C, D, H, W = y.shape
    n_elements = y.numel()
    S = D * H * W

    out = torch.empty_like(y)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    fused_post_conv_kernel[grid](
        y,
        bias,
        out,
        n_elements,
        S,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by
    fused elementwise post-processing using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return triton_fused_ops(x, self.bias)