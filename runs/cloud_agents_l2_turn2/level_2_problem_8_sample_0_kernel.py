import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_pool_avg_kernel_2x2x2(
    x_ptr,
    out_ptr,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    N,
    C,
    D_pool,
    H_pool,
    W_pool,
    num_pooled,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N * C:
        return

    n = pid // C
    c = pid % C

    channel_ptr = x_ptr + n * stride_n + c * stride_c

    acc = 0.0
    for block_start in range(0, num_pooled, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_pooled

        wp = offsets % W_pool
        hp = (offsets // W_pool) % H_pool
        dp = offsets // (W_pool * H_pool)

        d0 = dp * 2
        h0 = hp * 2
        w0 = wp * 2

        base = channel_ptr + d0 * stride_d + h0 * stride_h + w0 * stride_w

        v0 = tl.load(base, mask=mask, other=-float("inf"))
        v1 = tl.load(base + stride_w, mask=mask, other=-float("inf"))
        v2 = tl.load(base + stride_h, mask=mask, other=-float("inf"))
        v3 = tl.load(base + stride_h + stride_w, mask=mask, other=-float("inf"))
        v4 = tl.load(base + stride_d, mask=mask, other=-float("inf"))
        v5 = tl.load(base + stride_d + stride_w, mask=mask, other=-float("inf"))
        v6 = tl.load(base + stride_d + stride_h, mask=mask, other=-float("inf"))
        v7 = tl.load(base + stride_d + stride_h + stride_w, mask=mask, other=-float("inf"))

        m01 = tl.maximum(v0, v1)
        m23 = tl.maximum(v2, v3)
        m45 = tl.maximum(v4, v5)
        m67 = tl.maximum(v6, v7)
        m03 = tl.maximum(m01, m23)
        m47 = tl.maximum(m45, m67)
        m = tl.maximum(m03, m47)

        acc += tl.sum(tl.where(mask, m, 0.0), axis=0)

    res = (acc / divisor) / num_pooled
    tl.store(out_ptr + pid, res)


def fused_pool_global_avg(x: torch.Tensor, pool_size, divisor: float):
    N, C, D, H, W = x.shape
    pd, ph, pw = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size, pool_size)

    if pd == 2 and ph == 2 and pw == 2 and divisor > 0:
        D_pool = D // 2
        H_pool = H // 2
        W_pool = W // 2
        num_pooled = D_pool * H_pool * W_pool

        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=torch.float32)

        BLOCK_SIZE = 512
        grid = (N * C,)
        fused_pool_avg_kernel_2x2x2[grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            N,
            C,
            D_pool,
            H_pool,
            W_pool,
            num_pooled,
            float(divisor),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    else:
        x = x / divisor
        x = torch.nn.functional.max_pool3d(x, pool_size)
        x = torch.nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        return x


class ModelNew(nn.Module):
    """
    Optimized Model that performs 3D convolution followed by a fused
    divisor + max pooling + global average pooling Triton kernel,
    bias addition, and dimension reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = fused_pool_global_avg(x, self.pool_size, self.divisor)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x