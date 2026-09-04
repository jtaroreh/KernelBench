import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_k4s4_kernel(
    x_ptr,
    out_ptr,
    n_spatial,
    w_out,
    h_out,
    d_out,
    channels,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_bc = tl.program_id(1)

    b = pid_bc // channels
    c = pid_bc % channels

    x_bc_ptr = x_ptr + b * stride_b + c * stride_c
    out_bc_ptr = out_ptr + pid_bc * n_spatial

    spatial_offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < n_spatial

    ow = spatial_offsets % w_out
    rem = spatial_offsets // w_out
    oh = rem % h_out
    od = rem // h_out

    in_base = (od * 4) * stride_d + (oh * 4) * stride_h + (ow * 4) * stride_w

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for kd in range(4):
        for kh in range(4):
            for kw in range(4):
                offset = in_base + kd * stride_d + kh * stride_h + kw * stride_w
                val = tl.load(x_bc_ptr + offset, mask=mask, other=0.0)
                acc += val.to(tl.float32)

    out_val = acc * 0.015625
    tl.store(out_bc_ptr + spatial_offsets, out_val, mask=mask)


def triton_avg_pool3d_k4s4(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    B, C, D, H, W = x.shape
    d_out = D // 4
    h_out = H // 4
    w_out = W // 4
    n_spatial = d_out * h_out * w_out

    out = torch.empty((B, C, d_out, h_out, w_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_spatial, BLOCK_SIZE), B * C)

    avg_pool3d_k4s4_kernel[grid](
        x,
        out,
        n_spatial,
        w_out,
        h_out,
        d_out,
        C,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization,
    and a fused 3D average pooling layer (equivalent to two sequential 2x2x2 AvgPool3d layers).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        return triton_avg_pool3d_k4s4(x)
