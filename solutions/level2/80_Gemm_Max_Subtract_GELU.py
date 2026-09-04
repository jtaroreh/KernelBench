import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _exact_gelu(x):
    z = x * 0.7071067811865475
    abs_z = tl.abs(z)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * abs_z)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    exp_term = tl.exp(tl.clamp(-abs_z * abs_z, -88.0, 88.0))
    erf_approx = 1.0 - poly * exp_term
    erf_val = tl.where(z >= 0.0, erf_approx, -erf_approx)
    return 0.5 * x * (1.0 + erf_val)


@triton.jit
def fused_sub_gelu_kernel(
    vals_ptr,
    mean_ptr,
    out_ptr,
    n_elements,
    mean_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    v = tl.load(vals_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mean_ptr + (offsets % mean_numel), mask=mask, other=0.0).to(tl.float32)
    subbed = v - m
    res = _exact_gelu(subbed)
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, max operation, and fuses subtraction and GELU
    activation into a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        x = self.gemm(x)
        vals = torch.max(x, dim=self.max_dim, keepdim=True).values
        mean_vals = vals.mean(dim=1, keepdim=True)
        out = torch.empty_like(vals)
        n_elements = vals.numel()
        mean_numel = mean_vals.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_sub_gelu_kernel[grid](
            vals,
            mean_vals,
            out,
            n_elements,
            mean_numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, max_dim]
