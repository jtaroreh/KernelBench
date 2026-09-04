import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    N,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_x_ptr = x_ptr + row_idx * stride_xm
    row_out_ptr = out_ptr + row_idx * stride_outm

    # Pass 1: find max and sum(exp(x - max))
    row_max = -float("inf")
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    sum_exp = 0.0
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
        exp_x = tl.where(mask, tl.exp(tl.clamp(x - row_max, -88.0, 0.0)), 0.0)
        sum_exp += tl.sum(exp_x, axis=0)

    # Pass 2: write out exp(x - max) / sum_exp
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
        out_val = tl.exp(tl.clamp(x - row_max, -88.0, 0.0)) / sum_exp
        tl.store(row_out_ptr + cols * stride_outn, out_val, mask=mask)


def triton_fused_softmax(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 2048
    grid = (M,)
    fused_softmax_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        N,
        BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model performing matrix multiplication, dropout, and custom Triton row-softmax.
    """

    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.matmul(x)
        x = self.dropout(x)
        return triton_fused_softmax(x)


batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
