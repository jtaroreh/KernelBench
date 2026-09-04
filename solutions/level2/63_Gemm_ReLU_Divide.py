import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_relu_div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0) / divisor
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM via nn.Linear (cuBLAS) and fuses ReLU with division in a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        n_elements = x.numel()
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_relu_div_kernel[grid](
            x,
            out,
            n_elements,
            float(self.divisor),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
