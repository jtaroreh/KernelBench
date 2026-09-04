import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sigmoid_scale_residual_kernel(
    x_ptr,
    out_ptr,
    scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(val)
    res = sig * scaling_factor + val
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    Uses cuBLAS for GEMM and a custom Triton kernel fusing sigmoid, scaling, and residual addition.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_sigmoid_scale_residual_kernel[grid](
            x,
            out,
            float(self.scaling_factor),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
