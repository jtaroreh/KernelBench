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
def fused_gelu_softmax_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_x_ptr = x_ptr + row_idx * stride_xm
    row_out_ptr = out_ptr + row_idx * stride_outm

    # Pass 1: Find row max of _exact_gelu(x) across chunks of BLOCK_N
    row_max = -float("inf")
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        gelu_x = _exact_gelu(x)
        gelu_x = tl.where(mask, gelu_x, -float("inf"))
        row_max = tl.maximum(row_max, tl.max(gelu_x, axis=0))

    # Pass 2: Compute sum of exp(clamp(_exact_gelu(x) - row_max, -88.0, 88.0))
    sum_exp = 0.0
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        gelu_x = _exact_gelu(x)
        exp_val = tl.exp(tl.clamp(gelu_x - row_max, -88.0, 88.0))
        sum_exp += tl.sum(tl.where(mask, exp_val, 0.0), axis=0)

    # Pass 3: Store exp(clamp(_exact_gelu(x) - row_max, -88.0, 88.0)) / sum_exp
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        gelu_x = _exact_gelu(x)
        out_val = tl.exp(tl.clamp(gelu_x - row_max, -88.0, 88.0)) / sum_exp
        tl.store(row_out_ptr + cols * stride_outn, out_val, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, followed by a fused
    exact GELU and Softmax Triton kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        if not x.is_contiguous():
            x = x.contiguous()
        M, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = 2048
        grid = (M,)
        fused_gelu_softmax_kernel[grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            M,
            N,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
