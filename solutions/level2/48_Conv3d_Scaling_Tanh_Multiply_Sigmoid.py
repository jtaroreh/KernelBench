import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_tanh_mul_sig_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c_idx = (offsets // spatial_size) % C

    s_val = tl.load(scale_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    b_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    scaled = val * s_val
    tanh_val = 2.0 * tl.sigmoid(2.0 * scaled) - 1.0
    res = tl.sigmoid(tanh_val * b_val)

    tl.store(out_ptr + offsets, res, mask=mask)


def triton_fused_scale_tanh_mul_sig(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    B, C, D, H, W = x.shape
    spatial_size = D * H * W
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    fused_scale_tanh_mul_sig_kernel[grid](
        x,
        scale,
        bias,
        out,
        n_elements,
        C,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model performing 3D convolution followed by a fused
    scaling, tanh, bias multiplication, and sigmoid activation Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return triton_fused_scale_tanh_mul_sig(x, self.scaling_factor, self.bias)
