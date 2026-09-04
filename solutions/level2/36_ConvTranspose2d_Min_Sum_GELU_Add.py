import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _exact_gelu(x):
    z = x * 0.7071067811865475
    abs_z = tl.abs(z)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * abs_z)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    exp_term = tl.exp(tl.clamp(-abs_z * abs_z, -88.0, 88.0))
    erf_approx = 1.0 - poly * exp_term
    erf_val = tl.where(z >= 0.0, erf_approx, -erf_approx)
    return 0.5 * x * (1.0 + erf_val)


@triton.jit
def fused_gelu_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    bias_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + (offsets % bias_numel), mask=mask, other=0.0).to(tl.float32)
    res = _exact_gelu(val) + b
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation along channel dimension,
    sum operation along height dimension, and a fused GELU activation and bias addition in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = torch.sum(x, dim=2, keepdim=True)
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        bias_numel = self.bias.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_gelu_bias_kernel[grid](
            x,
            self.bias,
            out,
            n_elements,
            bias_numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
