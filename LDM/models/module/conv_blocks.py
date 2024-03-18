import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, stride=1, padding=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x     
    
class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 res_hidden=None,
                 time_emb_dim=None, groups=8):
        super().__init__()
        
        self.mlp = None
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), 
                          nn.Linear(time_emb_dim, out_channels * 2)
                          )

        if not res_hidden:
            res_hidden = out_channels
        
        self.block1 = ConvBlock(in_channels, res_hidden, groups=groups)
        self.block2 = ConvBlock(res_hidden, out_channels, groups=groups)
                
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb = None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            #time_emb = rearrange(time_emb, "b c -> b c 1 1")
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)     

def Upsample(dim, dim_out = None, conv = 'bilinear', scale = 2):
    if 'conv' in conv:
        return UpsampleConv(dim, dim_out)
    elif conv in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
        return UpsampleInterp(dim, dim_out, conv, scale)
    else:
        raise ValueError(f"Expected conv ot be 'conv', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', got '{conv}'")
    
    
def UpsampleInterp(dim, dim_out = None, interp = 'linear', scale = 2, def_align_corners = False):
    assert scale is not None and scale != 0, f'Scale must be specified!'
    if not dim_out:
            dim_out = dim
    if interp in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        align_corners = def_align_corners # check if helps against checkerboard pattern, was: True
    else:
        align_corners = None
    #align_corners = None # check if helps against checkerboard pattern
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode=interp, align_corners=align_corners),
        nn.Conv2d(in_channels=dim,
                  out_channels=dim_out, 
                  kernel_size=3, 
                  padding=1),
        nn.GroupNorm(max(1, dim_out//4), dim_out)
        )

def UpsampleConv(dim, dim_out=None):
    convT_kernel = 4
    convT_stride = 2
    convT_padding = 1
    if not dim_out:
        dim_out = dim
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=dim,
                           out_channels=dim_out,
                           kernel_size=convT_kernel, 
                           stride=convT_stride, 
                           padding=convT_padding),
        nn.GroupNorm(max(1, dim_out//4), dim_out)
    )

def Downsample(dim, dim_out = None, mode = 'avg', kern = 2):
    if 'conv' in mode:
        return DownsampleConv(dim, dim_out)
    else:
        assert kern is not None and kern != 0, f'Kernel size must be specified!'
        return DownsamplePool(dim, dim_out, mode, kern)


def DownsamplePool(dim, dim_out=None, mode = 'avg', kern = 2):
    if 'avg' in mode or 'mean' in mode:
        pooling = nn.AvgPool2d(kernel_size = kern)
    elif 'max' in mode:
        pooling = nn.MaxPool2d(kernel_size=kern)
    else:
        raise ValueError(f"Expected mode to be 'avg'/'mean' or 'max', got {mode} instead")
    if not dim_out:
        dim_out = dim
    return nn.Sequential(
            pooling, 
            nn.Conv2d(in_channels=dim,
                  out_channels=dim_out, 
                  kernel_size=3, 
                  padding=1),
        nn.GroupNorm(max(1, dim_out//4), dim_out)
    )


def DownsampleConv(dim, dim_out=None):
    conv_kernel = 3
    stride = 2
    padding = 1
    if not dim_out:
        dim_out = dim
    return nn.Sequential(
                nn.Conv2d(in_channels=dim, 
                          out_channels=dim_out,
                          kernel_size=conv_kernel,
                          stride=stride, padding=padding),
                nn.GroupNorm(max(1, dim_out//4), dim_out))
