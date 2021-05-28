import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization"""
    def forward(self, x):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], 
                        keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + 1e-5)
        out = F.conv2d(x, weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)
        return out

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block"""

    def __init__(self, cin, cout, cmid, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        eps = 1e-6
        self.stride = stride
        self.cin = cin
        self.cout = cout
        
        if self.stride != 1 or self.cin != self.cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.groupnorm_proj = nn.GroupNorm(cout, cout)
        
        self.groupnorm1 = nn.GroupNorm(32, cmid, eps=eps)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.groupnorm2 = nn.GroupNorm(32, cmid, eps=eps)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.groupnorm3 = nn.GroupNorm(32, cout, eps=eps)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        if self.stride != 1 or self.cin != self.cout:
            residual = self.downsample(x)
            residual = self.groupnorm_proj(residual)
        
        x = self.relu(self.groupnorm1(self.conv1(x)))
        x = self.relu(self.groupnorm2(self.conv2(x)))
        x = self.groupnorm3(self.conv3(x))
        x = self.relu(residual + x)
        return x

class ResNetv2(nn.Module):
    def __init__(self, blocks, width_factor=1):
        super().__init__()
        self.width = 64 * width_factor

        # Root block
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, self.width, kernel_size=7, stride=2, bias=False, padding=3)), # [b, 64, 112, 112]
            ('groupnorm', nn.GroupNorm(32, self.width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)) # [b, 64, 55, 55]
        ]))

        # Body block
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=self.width, cout=self.width*4, cmid=self.width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=self.width*4, cout=self.width*4, cmid=self.width)) for i in range(2, blocks[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=self.width*4, cout=self.width*8, cmid=self.width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=self.width*8, cout=self.width*8, cmid=self.width*2)) for i in range(2, blocks[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=self.width*8, cout=self.width*16, cmid=self.width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=self.width*16, cout=self.width*16, cmid=self.width*4)) for i in range(2, blocks[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        x = self.root(x) # [b, 64, 55, 55]
        x = self.body(x) # [b, in_channels, 14, 14]
        return x
