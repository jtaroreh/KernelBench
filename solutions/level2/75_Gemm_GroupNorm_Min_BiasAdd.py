import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_broadcast_bias_kernel(
    min_x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row_idx = offsets // out_features
    col_idx = offsets % out_features

    m_val = tl.load(min_x_ptr + row_idx, mask=mask, other=0.0).to(tl.float32)
    b_val = tl.load(bias_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)

    res = m_val + b_val
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs GEMM (nn.Linear), Group Normalization, Minimum operation,
    and a fused 1D Triton broadcast bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        x = self.group_norm(x)
        min_x = torch.min(x, dim=1, keepdim=True)[0]  # Shape (M, 1)
        M = x.shape[0]
        out = torch.empty((M, self.out_features, 1, 1), dtype=x.dtype, device=x.device)
        n_elements = out.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_broadcast_bias_kernel[grid](
            min_x,
            self.bias,
            out,
            n_elements,
            self.out_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out.permute(2, 1, 0, 3)


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
