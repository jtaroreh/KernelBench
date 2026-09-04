import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _mish(x):
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.clamp(x, -88.0, 88.0))))
    return x * _tanh(sp)


@triton.jit
def fused_bias_ht_mish_kernel(x_ptr, bias_ptr, out_ptr, n_elements, out_features, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    b = tl.load(bias_ptr + (offsets % out_features), mask=mask, other=0.0).to(tl.float32)
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32) + b
    clamped = tl.clamp(val, -1.0, 1.0)
    res = _mish(clamped)
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM, followed by a fused BiasAdd + Hardtanh + Mish
    Triton kernel, and finishes with GroupNorm.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.out_features = out_features

    def forward(self, x):
        x = self.gemm(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_bias_ht_mish_kernel[grid](
            x, self.bias, out, n_elements, self.out_features, BLOCK_SIZE=BLOCK_SIZE
        )
        x = self.groupnorm(out)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
