import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_logsumexp_relu_kernel(
    x_ptr, out_ptr,
    batch_size, channels, spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location across all channels
    pid = tl.program_id(0)
    
    # Calculate offsets for the current spatial location
    # Input is (B, C, S), we want to reduce over C
    # We process one (B, S) index per program
    b_idx = pid // spatial_size
    s_idx = pid % spatial_size
    
    # Online LogSumExp
    m_i = -float('inf')
    s_i = 0.0
    
    for c in range(0, channels, BLOCK_SIZE):
        offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = offsets < channels
        
        # Load channel values for this (b, s)
        vals = tl.load(x_ptr + (b_idx * channels * spatial_size + offsets * spatial_size + s_idx), mask=mask, other=-float('inf'))
        
        # Update max
        m_next = tl.maximum(m_i, tl.max(vals, axis=0))
        # Update sum
        s_i = s_i * tl.exp(m_i - m_next) + tl.sum(tl.exp(vals - m_next), axis=0)
        m_i = m_next
        
    res = m_i + tl.log(s_i)
    # ReLU
    out = tl.where(res > 0.0, res, 0.0)
    
    tl.store(out_ptr + (b_idx * spatial_size + s_idx), out)

def triton_logsumexp_relu(x: torch.Tensor):
    # x shape: (B, C, D, H, W) -> flatten to (B, C, D*H*W)
    B, C, D, H, W = x.shape
    spatial_size = D * H * W
    x_flat = x.view(B, C, spatial_size)
    out = torch.empty((B, 1, D, H, W), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (B * spatial_size,)
    
    fused_logsumexp_relu_kernel[grid](
        x_flat, out,
        B, C, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        # Replace torch.logsumexp and torch.relu with fused kernel
        x = triton_logsumexp_relu(x)
        return x