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
def fused_gelu_scale_row_max_kernel(
    x_ptr,
    out_ptr,
    scale_factor,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    val = tl.load(x_ptr + pid * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    gelu_val = _exact_gelu(val)
    scaled_gelu = gelu_val * scale_factor
    scaled_gelu = tl.where(mask, scaled_gelu, -float("inf"))
    row_max = tl.max(scaled_gelu, axis=0)
    tl.store(out_ptr + pid, row_max)


class ModelNew(nn.Module):
    """
    Optimized model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max"
    with a fused Triton kernel for GELU, scaling, and row max-reduction.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        if not x.is_contiguous():
            x = x.contiguous()
        M, N = x.shape
        out = torch.empty((M,), dtype=x.dtype, device=x.device)
        BLOCK_SIZE = 512
        grid = (M,)
        fused_gelu_scale_row_max_kernel[grid](
            x,
            out,
            float(self.scale_factor),
            M,
            N,
            x.stride(0),
            x.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
