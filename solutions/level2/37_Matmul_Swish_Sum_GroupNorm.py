import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_swish_bias_kernel(
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

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + (offsets % out_features), mask=mask, other=0.0).to(tl.float32)
    swish = val * tl.sigmoid(val)
    res = swish + b
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication (cuBLAS), fuses Swish activation and bias addition
    in a custom 1D Triton kernel, and applies GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.matmul(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_swish_bias_kernel[grid](
            x,
            self.bias,
            out,
            n_elements,
            self.out_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        x = self.group_norm(out)
        return x
