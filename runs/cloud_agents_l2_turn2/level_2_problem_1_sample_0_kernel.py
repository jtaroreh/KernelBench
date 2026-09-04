import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_relu_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    spatial_size,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton JIT kernel fusing ReLU and channel bias addition:
        out = relu(x) + bias
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input element
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Fused ReLU activation
    relu_x = tl.maximum(x, 0.0)

    # Compute channel index in NCHW format:
    # offset = n * (num_channels * spatial_size) + c * spatial_size + spatial_offset
    channel_idx = (offsets // spatial_size) % num_channels

    # Load corresponding channel bias
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)

    # Elementwise addition
    out = relu_x + bias_val

    # Store result to output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_relu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Helper function wrapping fused_relu_bias_kernel.

    Args:
        x: Input tensor of shape (batch_size, out_channels, height, width) on CUDA.
        bias: Bias tensor of shape (out_channels, 1, 1) or (out_channels,) on CUDA.

    Returns:
        torch.Tensor: Result of relu(x) + bias.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    if x.ndim == 4:
        num_channels = x.shape[1]
        spatial_size = x.shape[2] * x.shape[3]
    elif x.ndim == 3:
        num_channels = x.shape[0]
        spatial_size = x.shape[1] * x.shape[2]
    elif x.ndim == 2:
        num_channels = x.shape[1]
        spatial_size = 1
    else:
        num_channels = 1
        spatial_size = 1

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    fused_relu_bias_kernel[grid](
        x,
        bias,
        out,
        n_elements,
        spatial_size,
        num_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Conv2d followed by Triton-fused ReLU and channel bias addition.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias_shape: tuple) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return triton_fused_relu_bias(x, self.bias)


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
