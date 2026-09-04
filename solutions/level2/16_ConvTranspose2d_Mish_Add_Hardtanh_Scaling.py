import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_mish_add_hardtanh_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    add_value_f32,
    scale_f32,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # --- Mish activation ---
    # Mish(x) = x * tanh(softplus(x))
    # Tanh(x) = 2 * sigmoid(2 * x) - 1
    # softplus(x) = log(1 + exp(x))
    x_clamped = tl.clamp(x, -88.0, 88.0)
    softplus_x = tl.log(1.0 + tl.exp(x_clamped))
    
    # Compute tanh(softplus_x) using sigmoid identity
    tanh_softplus_x = 2.0 * tl.sigmoid(2.0 * softplus_x) - 1.0
    mish_x = x * tanh_softplus_x

    # --- Add value ---
    x_after_add = mish_x + add_value_f32

    # --- Hardtanh activation (min_val=-1, max_val=1) ---
    x_after_hardtanh = tl.maximum(tl.minimum(x_after_add, 1.0), -1.0)

    # --- Scale (multiply) ---
    out = x_after_hardtanh * scale_f32

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_activation_ops(x: torch.Tensor, add_value: float, scale: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    fused_mish_add_hardtanh_mul_kernel[grid](
        x,
        out,
        n_elements,
        float(add_value),
        float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then a fused
    Mish activation, add, Hardtanh activation, and scaling using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = float(add_value)
        self.scale = float(scale)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_activation_ops(x, self.add_value, self.scale)
        return x