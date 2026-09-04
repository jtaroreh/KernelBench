import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale_idx = offsets % out_features
    scale_val = tl.load(scale_ptr + scale_idx, mask=mask, other=0.0).to(tl.float32)
    out = val * scale_val
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM, applies fused scaling via a Triton kernel,
    and then applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        out_features = x.shape[-1]
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_scale_kernel[grid](
            x,
            self.scale,
            out,
            n_elements,
            out_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        x = self.bn(out)
        return x
