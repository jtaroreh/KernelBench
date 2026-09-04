import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    n_elements, n_spatial,
    clamp_min, clamp_max,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel of one batch (a spatial slice)
    pid = tl.program_id(0)
    
    # Pointers to the start of the spatial slice
    x_ptr += pid * n_spatial
    out_ptr += pid * n_spatial
    
    # Load scale for this channel
    scale = tl.load(scale_ptr + (pid % tl.num_programs(0)))
    
    # Online Softmax: 1st pass (max)
    m_i = -float('inf')
    for i in range(0, n_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        x = tl.maximum(clamp_min, tl.minimum(clamp_max, x))
        m_i = tl.maximum(m_i, tl.max(x, axis=0))
    
    # Online Softmax: 2nd pass (sum)
    l_i = 0.0
    for i in range(0, n_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        x = tl.maximum(clamp_min, tl.minimum(clamp_max, x))
        l_i += tl.sum(tl.exp(x - m_i), axis=0)
    
    # Online Softmax: 3rd pass (write)
    for i in range(0, n_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        x = tl.maximum(clamp_min, tl.minimum(clamp_max, x))
        out = (tl.exp(x - m_i) / l_i) * scale
        tl.store(out_ptr + offsets, out, mask=mask)

def triton_clamp_softmax_scale(x: torch.Tensor, scale: torch.Tensor, clamp_min: float, clamp_max: float):
    b, c, d, h, w = x.shape
    n_spatial = d * h * w
    out = torch.empty_like(x)
    
    grid = (b * c,)
    BLOCK_SIZE = 1024
    
    clamp_softmax_scale_kernel[grid](
        x, scale, out,
        x.numel(), n_spatial,
        clamp_min, clamp_max,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        # Fused clamp + softmax + scale kernel
        return triton_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)