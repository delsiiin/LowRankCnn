import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

class UpsampleFP32(nn.Module):
    def __init__(self, scale, dtype=torch.float32):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
    
    def forward(self, x):
        x_type = x.dtype
        if x.dtype != self.dtype and ((torch.get_autocast_gpu_dtype() == torch.bfloat16) or (x.dtype == torch.bfloat16)):
            x = x.to(self.dtype)
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if x_type != x.dtype:
            x = x.to(x_type)
        return x

def interpolate(x: torch.Tensor, size, interp_mode: str = None):
    if x.shape[-2:] == size: return x
    
    interp_mode = ('bilinear' if size[-1] >= x.shape[-1] else 'area') if interp_mode is None else interp_mode
    
    if torch.get_autocast_gpu_dtype() == torch.bfloat16: # F interpolate is not supported on bf16
        original_dtype = x.dtype
        with torch.autocast('cuda', torch.float32):
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            x = F.interpolate(x, size, mode=interp_mode)
        if x.dtype != original_dtype:
            x = x.to(original_dtype)
    else:
        x = F.interpolate(x, size, mode=interp_mode)
    
    return x

class ChannelSplit(nn.Module):
    def __init__(self, split):
        super().__init__()
        
        self.split = split
    
    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, H, self.split, W // self.split).permute(0, 1, 3, 2, 4).reshape(N, C*self.split, H, W//self.split)

class KeepRes(nn.Module):
    def __init__(self, *args, output_width=None):
        super().__init__()
        self.net = nn.Sequential(*args)
        self.output_width = output_width
    
    def forward(self, x):
        x_shape = x.shape
        x = self.net(x)
        if self.output_width is None:
            x = interpolate(x, x_shape[-2:])
        else:
            x = interpolate(x, (x_shape[-2], self.output_width))
        return x

class CausalConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 dilation: Tuple[int, int] = (1, 1),
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        conv_mask = torch.ones((1, 1, self.kernel_size[0], 1), dtype=torch.float32)
        self.register_buffer("conv_mask", conv_mask) 
        self.conv_mask[:, :, self.kernel_size[0] // 2 + 1 :, :] = 0.0
    
    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode=self.padding_mode),
                            weight, bias, self.stride,
                            0, self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.conv_mask
        return self._conv_forward(input, self.weight, self.bias)
    