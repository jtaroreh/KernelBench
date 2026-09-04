import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_bn_avgpool_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_D, out_H, out_W = D // 4, H // 4, W // 4
    total_out_elements = N * C * out_D * out_H * out_W
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out_elements
    
    # Map output index to (n, c, d_out, h_out, w_out)
    n = (offsets // (C * out_D * out_H * out_W))
    c = (offsets // (out_D * out_H * out_W)) % C
    d_out = (offsets // (out_H * out_W)) % out_D
    h_out = (offsets // out_W) % out_H
    w_out = offsets % out_W
    
    # Load BN parameters
    mean = tl.load(mean_ptr + c, mask=mask)
    var = tl.load(var_ptr + c, mask=mask)
    weight = tl.load(weight_ptr + c, mask=mask)
    bias = tl.load(bias_ptr + c, mask=mask)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Perform 4x4x4 average pooling
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                d_in = d_out * 4 + i
                h_in = h_out * 4 + j
                w_in = w_out * 4 + k
                idx = (((n * C + c) * D + d_in) * H + h_in) * W + w_in
                val = tl.load(x_ptr + idx, mask=mask)
                # Apply BN
                val = (val - mean) * inv_std * weight + bias
                sum_val += val
                
    tl.store(out_ptr + offsets, sum_val / 64.0, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        
        # The kernel fuses the two 2x2x2 AvgPools into one 4x4x4 window
        # We use the running stats from the BN layer
        mean = self.batch_norm.running_mean
        var = self.batch_norm.running_var
        weight = self.batch_norm.weight
        bias = self.batch_norm.bias
        
        N, C, D, H, W = x.shape
        out_shape = (N, C, D // 4, H // 4, W // 4)
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        
        n_elements = N * C * (D // 4) * (H // 4) * (W // 4)
        BLOCK_SIZE = 128
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        fused_bn_avgpool_kernel[(grid,)](
            x, mean, var, weight, bias, out,
            N, C, D, H, W,
            self.batch_norm.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out