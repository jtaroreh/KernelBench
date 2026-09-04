import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_clamp_mul_max_kernel(
    x_ptr,
    multiplier_ptr,
    out_ptr,
    total_elements,
    S,
    clamp_min,
    clamp_max,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    n = offsets // S
    rem = offsets % S
    base = n * (C * S) + rem

    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)

    for c in range(C):
        m = tl.load(multiplier_ptr + c)
        x_val = tl.load(x_ptr + base + c * S, mask=mask, other=0.0)
        clamped = tl.minimum(tl.maximum(x_val, clamp_min), clamp_max)
        val = clamped * m
        max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_fused_clamp_mul_max(
    x: torch.Tensor,
    multiplier: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    x = x.contiguous()
    multiplier = multiplier.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    total_elements = N * S
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    fused_clamp_mul_max_kernel[grid](
        x,
        multiplier,
        out,
        total_elements,
        S,
        float(clamp_min),
        float(clamp_max),
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Fused Triton implementation of Conv3d -> Multiply -> InstanceNorm3d -> Clamp -> Multiply -> Max.
    Epilogue fusing clamp, multiplier, and max reduction along channel dimension (dim=1).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        multiplier_shape,
        clamp_min,
        clamp_max,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = triton_fused_clamp_mul_max(
            x, self.multiplier, self.clamp_min, self.clamp_max
        )
        return x