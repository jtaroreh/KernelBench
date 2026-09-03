import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_bias_pool_kernel(
    x_ptr, out_ptr, bias_ptr,
    batch, channels, height, width,
    out_h, out_w,
    scaling_factor,
    pool_size: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Grid: (batch * channels, out_h, out_w)
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Calculate base pointers
    b = pid_bc // channels
    c = pid_bc % channels
    
    # Load bias
    bias = tl.load(bias_ptr + c)

    # Max pooling logic
    max_val = -float('inf')
    
    # Iterate over pool window
    for ph in range(pool_size):
        for pw in range(pool_size):
            h_in = pid_h * pool_size + ph
            w_in = pid_w * pool_size + pw
            
            idx = ((b * channels + c) * height + h_in) * width + w_in
            val = tl.load(x_ptr + idx)
            
            # Apply Tanh, Scaling, and Bias
            val = tl.tanh(val) * scaling_factor + bias
            
            max_val = tl.maximum(max_val, val)

    # Store result
    out_idx = ((b * channels + c) * out_h + pid_h) * out_w + pid_w
    tl.store(out_ptr + out_idx, max_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        
        batch, channels, h, w = x.shape
        out_h, out_w = h // self.pool_size, w // self.pool_size
        out = torch.empty((batch, channels, out_h, out_w), device=x.device)
        
        grid = (batch * channels, out_h, out_w)
        
        fused_activation_bias_pool_kernel[grid](
            x, out, self.bias.squeeze(),
            batch, channels, h, w,
            out_h, out_w,
            self.scaling_factor,
            pool_size=self.pool_size,
            BLOCK_H=1, BLOCK_W=1
        )
        return out