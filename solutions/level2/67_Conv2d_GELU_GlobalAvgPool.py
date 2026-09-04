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
def fused_gelu_global_avgpool_kernel(
    x_ptr,
    out_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    slice_idx = tl.program_id(0)
    acc = 0.0
    for offset in range(0, spatial_size, BLOCK_SIZE):
        chunk_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offsets < spatial_size
        val = tl.load(x_ptr + slice_idx * spatial_size + chunk_offsets, mask=mask, other=0.0)
        acc += tl.sum(_exact_gelu(val), axis=0)
    tl.store(out_ptr + slice_idx, acc / spatial_size)


class ModelNew(nn.Module):
    """
    Optimized model that performs Conv2d followed by Triton-fused GELU and global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        spatial_size = H * W
        out = torch.empty((B, C), dtype=x.dtype, device=x.device)

        BLOCK_SIZE = 2048
        grid = (B * C,)
        fused_gelu_global_avgpool_kernel[grid](
            x,
            out,
            spatial_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
