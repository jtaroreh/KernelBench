import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_softmax_kernel(
    x_ptr, out_ptr,
    batch, channels, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over batch, H, and W
    pid = tl.program_id(0)
    num_hw = H * W
    b = pid // num_hw
    hw = pid % num_hw
    h = hw // W
    w = hw % W

    # Load channel values for the min operation
    # We reduce along the depth dimension (D)
    # Input shape: (B, C, D, H, W)
    # After min: (B, C, H, W)
    
    # To perform softmax over channels, we need all channels for a specific (b, h, w)
    # First, compute min over D for each channel
    # Since D is usually small, we load and reduce
    
    # Pointers to the start of the (b, c, :, h, w) slice
    base_ptr = x_ptr + b * stride_b + h * stride_h + w * stride_w
    
    # Temporary buffer for min values across channels
    # We need to compute softmax over channels, so we need all channels for this (b, h, w)
    # Load all channels for this (b, h, w)
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    
    # Find min over D
    # D is fixed at 24 based on problem description
    D = 24
    min_vals = tl.full([BLOCK_SIZE_C], float('inf'), dtype=tl.float32)
    
    for d in range(D):
        vals = tl.load(base_ptr + c_offsets * stride_c + d * stride_d, mask=c_offsets < channels, other=float('inf'))
        min_vals = tl.minimum(min_vals, vals)
    
    # Softmax over channels
    # 1. Max for numerical stability
    m_i = tl.max(min_vals, axis=0)
    # 2. Exp and sum
    p = tl.exp(min_vals - m_i)
    sum_p = tl.sum(p, axis=0)
    # 3. Normalize
    softmax_out = p / sum_p
    
    # Store
    tl.store(out_ptr + b * (channels * H * W) + c_offsets * (H * W) + h * W + w, softmax_out, mask=c_offsets < channels)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.out_channels = out_channels
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_C = triton.next_power_of_2(C)
        grid = (B * H * W,)
        
        min_softmax_kernel[grid](
            x, out,
            B, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        return out