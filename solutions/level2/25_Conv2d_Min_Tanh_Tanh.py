import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_min_tanh_tanh_kernel(
    x_ptr,
    out_ptr,
    C,
    S,
    total_spatial,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offsets = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offsets < total_spatial

    b_idx = s_offsets // S
    s_idx = s_offsets % S
    base_offsets = b_idx * (C * S) + s_idx

    min_val = tl.full([BLOCK_S], float("inf"), dtype=tl.float32)

    for c in range(C):
        c_offsets = base_offsets + c * S
        vals = tl.load(x_ptr + c_offsets, mask=mask, other=float("inf")).to(tl.float32)
        min_val = tl.minimum(min_val, vals)

    # Double tanh via sigmoid identity
    t1 = 2.0 * tl.sigmoid(tl.clamp(2.0 * min_val, -20.0, 20.0)) - 1.0
    t2 = 2.0 * tl.sigmoid(tl.clamp(2.0 * t1, -20.0, 20.0)) - 1.0

    out_offsets = b_idx * S + s_idx
    tl.store(out_ptr + out_offsets, t2, mask=mask)


def triton_min_tanh_tanh(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    S = H * W
    total_spatial = B * S
    out = torch.empty((B, 1, H, W), device=x.device, dtype=x.dtype)

    BLOCK_S = 1024
    grid = (triton.cdiv(total_spatial, BLOCK_S),)

    fused_min_tanh_tanh_kernel[grid](
        x,
        out,
        C,
        S,
        total_spatial,
        BLOCK_S=BLOCK_S,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return triton_min_tanh_tanh(x.contiguous())


batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]