import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_tanh_hs_res_kernel(
    norm_ptr,
    conv_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    norm_val = tl.load(norm_ptr + offsets, mask=mask)
    t = 2.0 * tl.sigmoid(2.0 * norm_val) - 1.0
    hs = t * tl.clamp(t + 3.0, 0.0, 6.0) / 6.0
    conv_val = tl.load(conv_ptr + offsets, mask=mask)
    res = conv_val + hs
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, Group Normalization,
    fused Tanh + HardSwish + Residual Addition with Triton, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        out = torch.empty_like(x_conv)
        n_elements = x_conv.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_tanh_hs_res_kernel[grid](
            x_norm, x_conv, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        x_logsumexp = torch.logsumexp(out, dim=1, keepdim=True)
        return x_logsumexp


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
