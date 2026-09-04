import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_softmax_kernel(
    x_ptr, out_ptr,
    batch, channels, H, W, D,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_hw = H * W
    b = pid // num_hw
    hw = pid % num_hw
    h = hw // W
    w = hw % W

    # Load channel values for the min operation
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    mask_c = c_offsets < channels
    
    # Pointers to the start of the (b, :, :, h, w) slice
    base_ptr = x_ptr + b * stride_b + h * stride_h + w * stride_w
    
    # Find min over D
    min_vals = tl.full([BLOCK_SIZE_C], float('inf'), dtype=tl.float32)
    
    for d in range(D):
        vals = tl.load(base_ptr + c_offsets * stride_c + d * stride_d, mask=mask_c, other=float('inf'))
        min_vals = tl.minimum(min_vals, vals)
    
    # Softmax over channels
    # 1. Max for numerical stability
    m_i = tl.max(min_vals, axis=0)
    # 2. Exp and sum (clamp to prevent overflow)
    p = tl.exp(tl.clamp(min_vals - m_i, -88.0, 88.0))
    # Mask out invalid channels for sum
    p = tl.where(mask_c, p, 0.0)
    sum_p = tl.sum(p, axis=0)
    
    # 3. Normalize
    softmax_out = p / sum_p
    
    # Store
    tl.store(out_ptr + b * (channels * H * W) + c_offsets * (H * W) + h * W + w, softmax_out, mask=mask_c)

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
        out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE_C = triton.next_power_of_2(C)
        grid = (B * H * W,)
        
        min_softmax_kernel[grid](
            x, out,
            B, C, H, W, D,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        return out