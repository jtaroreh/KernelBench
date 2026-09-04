import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_residual_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * (1.0 + scaling_factor)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale_residual(x: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    fused_scale_residual_kernel[grid](
        x,
        out,
        n_elements,
        float(scaling_factor),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model performing matrix multiplication followed by a fused
    scaling and residual addition Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.matmul(x)
        return triton_scale_residual(x, self.scaling_factor)
