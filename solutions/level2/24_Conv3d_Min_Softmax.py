import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def channel_softmax_kernel(
    x_ptr, out_ptr,
    batch, channels, H, W,
    stride_b, stride_c, stride_h, stride_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    hw = pid % (H * W)
    b = pid // (H * W)
    h = hw // W
    w = hw % W

    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    mask_c = c_offsets < channels

    in_ptr = x_ptr + b * stride_b + c_offsets * stride_c + h * stride_h + w * stride_w
    vals = tl.load(in_ptr, mask=mask_c, other=-float('inf')).to(tl.float32)

    m = tl.max(vals, axis=0)
    p = tl.exp(tl.clamp(vals - m, -88.0, 88.0))
    p = tl.where(mask_c, p, 0.0)
    sum_p = tl.sum(p, axis=0)

    out = p / sum_p
    out_ptr_base = out_ptr + b * stride_out_b + c_offsets * stride_out_c + h * stride_out_h + w * stride_out_w
    tl.store(out_ptr_base, out, mask=mask_c)

def triton_channel_softmax(x: torch.Tensor):
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    grid = (B * H * W,)
    channel_softmax_kernel[grid](
        x, out,
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        x = torch.min(x, dim=self.dim)[0]
        return triton_channel_softmax(x)