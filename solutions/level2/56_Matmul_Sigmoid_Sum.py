import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_row_sum_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start_ptr = x_ptr + row_idx * stride_xm
    acc = 0.0
    for col_offset in range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        chunk = tl.load(row_start_ptr + cols * stride_xn, mask=mask, other=0.0)
        sig = tl.sigmoid(chunk.to(tl.float32))
        sig = tl.where(mask, sig, 0.0)
        acc += tl.sum(sig, axis=0)

    tl.store(out_ptr + row_idx * stride_om, acc)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with cuBLAS
    and fuses sigmoid activation with row sum reduction in a custom Triton kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)
        M, N = x.shape
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        BLOCK_N = 2048
        grid = (M,)
        fused_sigmoid_row_sum_kernel[grid](
            x,
            out,
            M,
            N,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
