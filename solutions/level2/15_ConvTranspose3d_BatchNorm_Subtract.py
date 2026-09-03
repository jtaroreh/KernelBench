import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def spatial_mean_sub_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    spatial_size,
    num_channels,
    HAS_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # slice index: 0 <= pid < N * C
    c = pid % num_channels
    slice_offset = pid * spatial_size

    # Pass 1: compute sum over spatial dimensions
    sum_val = 0.0
    for offset in range(0, spatial_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < spatial_size
        x = tl.load(x_ptr + slice_offset + cols, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean_val = sum_val / spatial_size

    if HAS_SCALE:
        scale = tl.load(scale_ptr + c)
    else:
        scale = 1.0

    # Pass 2: subtract spatial mean and apply scale
    for offset in range(0, spatial_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < spatial_size
        x = tl.load(x_ptr + slice_offset + cols, mask=mask, other=0.0)
        out = (x - mean_val) * scale
        tl.store(out_ptr + slice_offset + cols, out, mask=mask)


def fused_spatial_norm(x: torch.Tensor, scale: torch.Tensor = None):
    x = x.contiguous()
    out = torch.empty_like(x)

    N, C = x.shape[0], x.shape[1]
    spatial_size = x.numel() // (N * C)
    grid = (N * C,)

    BLOCK_SIZE = 2048
    has_scale = scale is not None
    if has_scale:
        scale = scale.contiguous()

    spatial_mean_sub_scale_kernel[grid](
        x,
        scale,
        out,
        spatial_size,
        C,
        HAS_SCALE=has_scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        if not self.training and self.batch_norm.running_var is not None:
            gamma = self.batch_norm.weight if self.batch_norm.weight is not None else 1.0
            scale = gamma / torch.sqrt(self.batch_norm.running_var + self.batch_norm.eps)
            return fused_spatial_norm(x, scale)
        else:
            x = self.batch_norm(x)
            return fused_spatial_norm(x, None)