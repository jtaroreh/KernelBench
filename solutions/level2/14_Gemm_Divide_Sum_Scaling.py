import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_row_sum_scale_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    factor,
    stride_xm,
    stride_xn,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = x_ptr + pid * stride_xm
    acc = 0.0
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        chunk = tl.load(row_ptr + cols * stride_xn, mask=mask, other=0.0)
        acc += tl.sum(chunk.to(tl.float32), axis=0)

    tl.store(out_ptr + pid * stride_om, acc * factor)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with cuBLAS
    and fuses division, summation, and scaling into a custom Triton row-reduction kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = x @ self.weight.T
        M, N = x.shape
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        factor = float(self.scaling_factor) * 0.5

        BLOCK_N = 2048
        grid = (M,)
        fused_row_sum_scale_kernel[grid](
            x,
            out,
            M,
            N,
            factor,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
