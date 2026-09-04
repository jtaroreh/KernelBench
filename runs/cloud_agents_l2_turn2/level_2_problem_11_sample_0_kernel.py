import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _batchnorm_tanh_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N_CHANNELS, N_H, N_W,
    EPS: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    mean_val = tl.load(mean_ptr + pid_c)
    var_val = tl.load(var_ptr + pid_c)
    weight_val = tl.load(weight_ptr + pid_c)
    bias_val = tl.load(bias_ptr + pid_c)

    rstd = 1.0 / tl.sqrt(var_val + EPS)
    num_hw_elements = N_H * N_W
    offsets_hw = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    mask_hw = offsets_hw < num_hw_elements
    
    x_base_ptr = x_ptr + (pid_n * N_CHANNELS + pid_c) * num_hw_elements
    out_base_ptr = out_ptr + (pid_n * N_CHANNELS + pid_c) * num_hw_elements

    x = tl.load(x_base_ptr + offsets_hw, mask=mask_hw, other=0.0)
    normed_x = (x - mean_val) * rstd
    scaled_shifted_x = normed_x * weight_val + bias_val
    # Use tl.math.tanh as tl.tanh does not exist
    out = tl.math.tanh(scaled_shifted_x)
    tl.store(out_base_ptr + offsets_hw, out, mask=mask_hw)

def triton_batchnorm_tanh(x: torch.Tensor, bn_module: nn.BatchNorm2d):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_HW = 256
    num_blocks_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid = (N, C, num_blocks_hw)
    _batchnorm_tanh_kernel[grid](
        x, bn_module.running_mean, bn_module.running_var, bn_module.weight, bn_module.bias, out,
        C, H, W, EPS=bn_module.eps, BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return out

@triton.jit
def _groupnorm_mean_var_kernel(
    x_ptr, mean_out_ptr, var_out_ptr, N_CHANNELS, N_H, N_W,
    NUM_GROUPS: tl.constexpr, CHANNELS_PER_GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    
    sum_val = 0.0
    sum_sq_val = 0.0
    num_elements = CHANNELS_PER_GROUP * N_H * N_W
    
    for c in range(pid_g * CHANNELS_PER_GROUP, (pid_g + 1) * CHANNELS_PER_GROUP):
        x_ptr_c = x_ptr + (pid_n * N_CHANNELS + c) * (N_H * N_W)
        for i in range(0, N_H * N_W, 1024):
            offsets = i + tl.arange(0, 1024)
            mask = offsets < (N_H * N_W)
            vals = tl.load(x_ptr_c + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(vals)
            sum_sq_val += tl.sum(vals * vals)
            
    mean = sum_val / num_elements
    var = (sum_sq_val / num_elements) - (mean * mean)
    tl.store(mean_out_ptr + pid_n * NUM_GROUPS + pid_g, mean)
    tl.store(var_out_ptr + pid_n * NUM_GROUPS + pid_g, var)

def triton_groupnorm_mean_var(x, num_groups):
    N, C, H, W = x.shape
    mean = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    var = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    _groupnorm_mean_var_kernel[(N, num_groups)](
        x, mean, var, C, H, W, NUM_GROUPS=num_groups, CHANNELS_PER_GROUP=C // num_groups
    )
    return mean, var

@triton.jit
def _groupnorm_normalize_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N_CHANNELS, N_H, N_W, NUM_GROUPS: tl.constexpr, CHANNELS_PER_GROUP: tl.constexpr, EPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    pid_g = pid_c // CHANNELS_PER_GROUP
    mean = tl.load(mean_ptr + pid_n * NUM_GROUPS + pid_g)
    var = tl.load(var_ptr + pid_n * NUM_GROUPS + pid_g)
    weight = tl.load(weight_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)
    
    rstd = 1.0 / tl.sqrt(var + EPS)
    offsets = pid_hw * 256 + tl.arange(0, 256)
    mask = offsets < (N_H * N_W)
    
    x = tl.load(x_ptr + (pid_n * N_CHANNELS + pid_c) * (N_H * N_W) + offsets, mask=mask, other=0.0)
    out = ((x - mean) * rstd) * weight + bias
    tl.store(out_ptr + (pid_n * N_CHANNELS + pid_c) * (N_H * N_W) + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_batchnorm_tanh(x, self.batch_norm)
        x = self.max_pool(x)
        mean, var = triton_groupnorm_mean_var(x, self.group_norm.num_groups)
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        _groupnorm_normalize_kernel[(N, C, (H * W + 255) // 256)](
            x, mean, var, self.group_norm.weight, self.group_norm.bias, out,
            C, H, W, NUM_GROUPS=self.group_norm.num_groups, CHANNELS_PER_GROUP=C // self.group_norm.num_groups, EPS=self.group_norm.eps
        )
        return out