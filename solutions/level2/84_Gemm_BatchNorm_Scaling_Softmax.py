import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_softmax_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    N,
    SCALE_IS_SCALAR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_x_ptr = x_ptr + pid * stride_xm
    row_out_ptr = out_ptr + pid * stride_outm

    if SCALE_IS_SCALAR:
        scale_val = tl.load(scale_ptr).to(tl.float32)
    else:
        scale_val = 0.0

    # Pass 1: find max across chunks of BLOCK_N = 2048 with scale applied: val = tl.load(...) * scale_val
    max_val = -float("inf")
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        if SCALE_IS_SCALAR:
            val = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32) * scale_val
        else:
            s_vec = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            val = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32) * s_vec
        val = tl.where(mask, val, -float("inf"))
        max_val = tl.maximum(max_val, tl.max(val, axis=0))

    # Pass 2: compute sum(exp(clamp(val - max_val, -88.0, 88.0)))
    sum_exp = 0.0
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        if SCALE_IS_SCALAR:
            val = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32) * scale_val
        else:
            s_vec = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            val = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32) * s_vec
        exp_val = tl.where(mask, tl.exp(tl.clamp(val - max_val, -88.0, 88.0)), 0.0)
        sum_exp += tl.sum(exp_val, axis=0)

    # Pass 3: store exp(clamp(val - max_val, -88.0, 88.0)) / sum_exp
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        if SCALE_IS_SCALAR:
            val = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32) * scale_val
        else:
            s_vec = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            val = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32) * s_vec
        out_val = tl.exp(tl.clamp(val - max_val, -88.0, 88.0)) / sum_exp
        tl.store(row_out_ptr + cols * stride_outn, out_val, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM, BatchNorm1d, and a fused Triton kernel for scaling and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        x = self.bn(x)
        if not x.is_contiguous():
            x = x.contiguous()
        M, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = 2048
        grid = (M,)
        scale_is_scalar = (self.scale.numel() == 1)
        scale_tensor = self.scale.contiguous()
        fused_scale_softmax_kernel[grid](
            x,
            scale_tensor,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            N,
            SCALE_IS_SCALAR=scale_is_scalar,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
