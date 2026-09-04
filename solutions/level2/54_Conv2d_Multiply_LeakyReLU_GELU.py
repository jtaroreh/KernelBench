import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_mult_lrelu_gelu_kernel(
    x_ptr,
    multiplier_ptr,
    out_ptr,
    n_elements,
    C,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c_idx = (offsets // spatial_size) % C
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    m_val = tl.load(multiplier_ptr + c_idx, mask=mask, other=0.0)

    val = x_val * m_val
    lrelu = tl.where(val >= 0.0, val, val * 0.01)

    # Exact GELU via Abramowitz & Stegun 7.1.26
    z = lrelu * 0.7071067811865475
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
    out_val = 0.5 * lrelu * (1.0 + erf_val)

    tl.store(out_ptr + offsets, out_val, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model performing Conv2d followed by a fused Multiply + LeakyReLU + GELU Triton kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

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
        fused_mult_lrelu_gelu_kernel[grid](
            x,
            self.multiplier,
            out,
            n_elements,
            C,
            spatial_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
