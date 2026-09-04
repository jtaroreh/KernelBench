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
def fused_activations_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    l1 = tl.where(val >= 0.0, val, val * 0.01)
    l2 = tl.where(l1 >= 0.0, l1, l1 * 0.01)
    g1 = _exact_gelu(l2)
    g2 = _exact_gelu(g1)
    tl.store(out_ptr + offsets, g2, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp,
    and a fused Triton kernel for LeakyReLU + LeakyReLU + GELU + GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_activations_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
