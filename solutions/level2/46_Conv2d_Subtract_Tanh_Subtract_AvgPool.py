import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sub1_tanh_sub2_kernel(
    x_ptr,
    out_ptr,
    subtract1_value,
    subtract2_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask)
    sub1 = val - subtract1_value
    t = 2.0 * tl.sigmoid(2.0 * sub1) - 1.0  # Tanh identity
    res = t - subtract2_value
    tl.store(out_ptr + offsets, res, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    Fuses the subtraction, tanh activation, and second subtraction into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_sub1_tanh_sub2_kernel[grid](
            x,
            out,
            self.subtract1_value,
            self.subtract2_value,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        x = self.avgpool(out)
        return x
