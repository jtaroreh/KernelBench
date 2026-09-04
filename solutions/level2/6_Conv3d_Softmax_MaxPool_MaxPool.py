import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_dim1_kernel(
    x_ptr,
    out_ptr,
    n_spatial,
    n_channels,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_spatial:
        return

    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    mask = c_offsets < n_channels

    offsets = pid * n_channels + c_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    numerator = tl.exp(tl.clamp(x_shifted, -88.0, 88.0))
    numerator = tl.where(mask, numerator, 0.0)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    tl.store(out_ptr + offsets, softmax_out, mask=mask)

def triton_softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    B, C, D, H, W = x.shape
    n_spatial = B * D * H * W
    x_perm = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    out_perm = torch.empty_like(x_perm)

    BLOCK_SIZE_C = triton.next_power_of_2(C)
    grid = (n_spatial,)
    softmax_dim1_kernel[grid](
        x_perm,
        out_perm,
        n_spatial,
        C,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out_perm.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_softmax_dim1(x)
        x = self.pool1(x)
        x = self.pool2(x)
        return x