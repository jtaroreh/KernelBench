import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_add_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    bias_val = tl.load(bias_ptr + (offsets % out_features), mask=mask, other=0.0)
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0) + bias_val
    res = tl.maximum(val, 0.0)
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM via nn.Linear and fuses bias addition
    and ReLU activation into a 1D Triton kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        out_features = self.gemm.out_features
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_add_relu_kernel[grid](
            x,
            self.bias,
            out,
            n_elements,
            out_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
