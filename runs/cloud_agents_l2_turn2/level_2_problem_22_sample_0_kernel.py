import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_matmul_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xk,
    stride_out,
    K,
    scale_factor,
    clamp_min,
    clamp_max,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Online logsumexp accumulators
    m = -float("inf")
    s = 0.0

    two_scale = 2.0 * scale_factor

    for k_start in range(0, K, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        mask = k_offs < K

        x_ptrs = x_ptr + row_idx * stride_xm + k_offs * stride_xk
        vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))

        # Scale and residual addition (x + x = 2x)
        scaled = vals * two_scale
        clamped = tl.minimum(tl.maximum(scaled, clamp_min), clamp_max)
        clamped = tl.where(mask, clamped, -float("inf"))

        # Online logsumexp reduction
        chunk_m = tl.max(clamped, axis=0)
        new_m = tl.maximum(m, chunk_m)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(clamped - new_m), axis=0)
        m = new_m

    lse = tl.log(s) + m

    # Mish activation: mish(lse) = lse * tanh(softplus(lse))
    # Softplus: log(1 + exp(lse))
    sp = tl.where(lse > 20.0, lse, tl.log(1.0 + tl.exp(lse)))
    exp2 = tl.exp(2.0 * sp)
    tanh_sp = tl.where(sp > 10.0, 1.0, (exp2 - 1.0) / (exp2 + 1.0))
    mish = lse * tanh_sp

    # Final result: lse * mish(lse)
    res = lse * mish

    tl.store(out_ptr + row_idx * stride_out, res)


class ModelNew(nn.Module):
    """
    Optimized Model that performs matrix multiplication followed by a fused
    Triton kernel for scaling, clamping, logsumexp reduction, and Mish activation.
    """

    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High-performance cuBLAS linear layer
        x_mm = F.linear(x, self.matmul.weight, self.matmul.bias)

        M, K = x_mm.shape
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        BLOCK_SIZE = min(8192, max(512, triton.next_power_of_2(K)))
        grid = (M,)

        fused_post_matmul_kernel[grid](
            x_mm,
            out,
            x_mm.stride(0),
            x_mm.stride(1),
            out.stride(0),
            K,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        return out