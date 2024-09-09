from torch import nn
import torch
from functools import partial

from models.module.blocks import (ResnetBlock,
                                Residual,
                                PreNorm,
                                ConvBlock,
                                MidBlock)
from models.module.attention import LinearAttention

def scale_tensor_11(tensor):
    tmin = tensor.min().item()
    tmax = tensor.max().item() 
    return (tensor - tmin)/(abs(tmax-tmin)) * 2 -1
  
class Decoder(nn.Module):
    def __init__(self, 
                 in_planes=4,
                 init_planes=32,
                 out_planes=3,
                 plains_divs=(8, 4, 2, 1), 
                 resnet_grnorm_groups=4,
                 resnet_stacks=2,
                 up_mode='bilinear',
                 scale=2,
                 attention=[],
                 attn_heads=4,
                 attn_dim=32,
                 eps=1e-6,
                 legacy_mid=False,
                 tanh_out=False):
        super().__init__()
        
        self.tanh = tanh_out
        
        dims = [init_planes * m for m in plains_divs]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.conv_in = ConvBlock(in_planes, dims[0], kernel_size=3, stride=1, padding=1)
        
        layers = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # Upsample if not the first layer
            if ind > 0:
                layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode=up_mode),
                    ConvBlock(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
                ))
            
            for _ in range(resnet_stacks):
                layers.append(ResnetBlock(dim_out, dim_out, groups=resnet_grnorm_groups))
            
            if dim_out in attention or ind in attention:
                layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim))))
        
        self.upscale = nn.Sequential(*layers)
        
        mid_layers = [MidBlock(dims[-1], dims[-1], t_emb_dim=attn_dim, num_heads=attn_heads, num_layers=resnet_stacks, attn=True, norm_channels=resnet_grnorm_groups) for _ in range(resnet_stacks)]
        if not legacy_mid:
            mid_layers.insert(len(mid_layers) // 2, Residual(PreNorm(dims[-1], LinearAttention(dims[-1], heads=attn_heads, dim_head=attn_dim))))
        self.mid_block = nn.Sequential(*mid_layers)
        
        self.post_up = nn.Sequential(
                            nn.GroupNorm(num_groups=resnet_grnorm_groups, num_channels=dims[-1], eps=eps),
                            nn.SiLU(),
                            ConvBlock(dims[-1], out_planes, kernel_size=3, stride=1, padding=1)
                            )
                
    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        x = self.upscale(x)
        x = self.post_up(x)
        if self.tanh:
            x = torch.tanh(x)
        return x