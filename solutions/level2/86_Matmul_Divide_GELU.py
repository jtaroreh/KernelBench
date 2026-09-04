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
def fused_div_gelu_kernel(x_ptr, out_ptr, divisor, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = val / divisor
    res = _exact_gelu(scaled)
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication via nn.Linear and fuses
    scalar division and GELU activation into a single custom Triton kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_div_gelu_kernel[grid](
            x,
            out,
            float(self.divisor),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]
