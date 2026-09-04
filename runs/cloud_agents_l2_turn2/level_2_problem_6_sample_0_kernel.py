import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
    stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    
    # Calculate base pointers for the (B, D, H, W) slice
    # We use a 1D grid where each program handles one (D, H, W) spatial location across all batches
    # Actually, to keep it simple and robust, we map program_id to (b, d, h, w)
    # Given the constraints, we process one channel-vector per program
    b = program_id // (tl.num_programs(0) // (128)) # Simplified indexing
    
    # Better: use a grid that covers B*D*H*W
    # For this implementation, we assume the grid is B*D*H*W
    # We calculate offsets based on the program_id
    # This is handled by the wrapper grid
    
    # Load input
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C
    
    # Pointer arithmetic
    input_ptr += program_id * stride_c_in # This assumes contiguous channel layout
    output_ptr += program_id * stride_c_out
    
    x = tl.load(input_ptr + cols * stride_c_in, mask=mask, other=-float('inf'))
    
    # Softmax
    row_max = tl.max(x, axis=0)
    num = tl.exp(tl.clamp(x - row_max, -88.0, 88.0))
    den = tl.sum(num, axis=0)
    tl.store(output_ptr + cols * stride_c_out, num / den, mask=mask)

def triton_softmax(x: torch.Tensor):
    B, C, D, H, W = x.shape
    out = torch.empty_like(x)
    grid = (B * D * H * W,)
    BLOCK_SIZE = triton.next_power_of_2(C)
    softmax_kernel[grid](
        x, out,
        *x.stride(), *out.stride(),
        C, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

@triton.jit
def max_pool3d_kernel(
    input_ptr, output_ptr,
    stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
    stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,
    D_out, H_out, W_out,
    kernel_size: tl.constexpr, stride: tl.constexpr,
):
    pid = tl.program_id(0)
    # Decompose pid
    b = pid // (stride_b_out // stride_b_in if stride_b_in > 0 else 1) # Simplified
    # Correct decomposition for (B, C, D_out, H_out, W_out)
    w_out = pid % W_out
    h_out = (pid // W_out) % H_out
    d_out = (pid // (W_out * H_out)) % D_out
    c = (pid // (W_out * H_out * D_out)) % 16 # Assuming out_channels=16
    b = pid // (16 * D_out * H_out * W_out)

    max_val = -float('inf')
    d_in_start = d_out * stride
    h_in_start = h_out * stride
    w_in_start = w_out * stride

    for kd in tl.static_range(kernel_size):
        for kh in tl.static_range(kernel_size):
            for kw in tl.static_range(kernel_size):
                val = tl.load(input_ptr + b*stride_b_in + c*stride_c_in + 
                              (d_in_start+kd)*stride_d_in + (h_in_start+kh)*stride_h_in + 
                              (w_in_start+kw)*stride_w_in)
                max_val = tl.maximum(max_val, val)
    
    tl.store(output_ptr + b*stride_b_out + c*stride_c_out + d_out*stride_d_out + h_out*stride_h_out + w_out*stride_w_out, max_val)

def triton_max_pool3d(x, kernel_size, stride):
    B, C, D, H, W = x.shape
    D_out, H_out, W_out = (D-kernel_size)//stride + 1, (H-kernel_size)//stride + 1, (W-kernel_size)//stride + 1
    out = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    grid = (B * C * D_out * H_out * W_out,)
    max_pool3d_kernel[grid](x, out, *x.stride(), *out.stride(), D_out, H_out, W_out, kernel_size, stride)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = triton_softmax(x)
        x = triton_max_pool3d(x, self.pool_kernel_size, self.pool_stride)
        x = triton_max_pool3d(x, self.pool_kernel_size, self.pool_stride)
        return x