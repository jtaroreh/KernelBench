import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_min_sub_kernel(
    x_ptr,
    out_ptr,
    constant,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.minimum(val, constant) - constant
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using PyTorch's nn.Linear (cuBLAS)
    followed by a fused Triton element-wise kernel for minimum and subtraction.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_min_sub_kernel[grid](
            x,
            out,
            float(self.constant.item()),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
