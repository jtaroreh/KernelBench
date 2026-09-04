import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gelu_group_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    H,
    W,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_groups = N * num_groups
    if pid >= total_groups:
        return

    n = pid // num_groups
    g = pid % num_groups

    channels_per_group = C // num_groups
    HW = H * W
    M = channels_per_group * HW

    group_offset = (n * C + g * channels_per_group) * HW
    x_group_ptr = x_ptr + group_offset
    out_group_ptr = out_ptr + group_offset

    # Pass 1: compute mean
    mean_accum = 0.0
    for off in range(0, M, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(x_group_ptr + offsets, mask=mask, other=0.0)
        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475))
        mean_accum += tl.sum(tl.where(mask, y, 0.0))

    mean = mean_accum / M

    # Pass 2: compute variance
    var_accum = 0.0
    for off in range(0, M, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(x_group_ptr + offsets, mask=mask, other=0.0)
        y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475))
        diff = tl.where(mask, y - mean, 0.0)
        var_accum += tl.sum(diff * diff)

    var = var_accum / M
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize, scale, bias, and store
    for off in range(0, M, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(x_group_ptr + offsets, mask=mask, other=0.0)
        y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475))

        local_c = offsets // HW
        global_c = g * channels_per_group + local_c

        if weight_ptr is not None:
            w = tl.load(weight_ptr + global_c, mask=mask, other=1.0)
        else:
            w = 1.0

        if bias_ptr is not None:
            b = tl.load(bias_ptr + global_c, mask=mask, other=0.0)
        else:
            b = 0.0

        out = (y - mean) * rstd * w + b
        tl.store(out_group_ptr + offsets, out, mask=mask)


def fused_gelu_group_norm(x: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total_groups = N * num_groups

    BLOCK_SIZE = 4096
    grid = (total_groups,)

    fused_gelu_group_norm_kernel[grid](
        x,
        out,
        weight,
        bias,
        N,
        C,
        H,
        W,
        num_groups,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_gelu_group_norm(
            x,
            num_groups=self.group_norm.num_groups,
            weight=self.group_norm.weight,
            bias=self.group_norm.bias,
            eps=self.group_norm.eps,
        )
        return x