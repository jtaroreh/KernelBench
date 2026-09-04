import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_matmul_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, out_features,
    divide_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load bias (broadcasted)
    bias = tl.load(bias_ptr)
    
    # Apply bias and division
    x = (x + bias) / divide_value
    
    # Swish activation: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x
    
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.out_features = out_features

    def forward(self, x):
        # Matmul and BN are kept as standard PyTorch ops for stability/performance
        x = self.matmul(x)
        x = self.bn(x)
        
        # Fused kernel for: x + bias, division, and Swish
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fused_post_matmul_kernel[grid](
            x, self.bias, out,
            n_elements, self.out_features,
            self.divide_value,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out