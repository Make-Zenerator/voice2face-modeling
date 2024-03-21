from torch import nn
from models.module.blocks import (ConvBlock,
                                ResnetBlock,
                                Residual,
                                PreNorm,
                                MidBlock)
from models.module.attention import LinearAttention


class Encoder(nn.Module):
    def __init__(self, 
                 in_planes=3,
                 init_planes=32, 
                 planes_mults=(1, 2, 4, 8), # 1 2 4 8 / 1 1 2 4 / 1 2 4 4
                 resnet_grnorm_groups=4,
                 resnet_stacks=2,
                 attention=[],
                 attn_heads=4,
                 attn_dim=32,
                 latent_dim=4,
                 eps=1e-6,
                 legacy_mid=False
                ):
        super().__init__()
        
        dims = [init_planes, *map(lambda m: init_planes * m, planes_mults)] 
        in_out_dims = list(zip(dims[:-1], dims[1:]))
        
        self.init_conv = ConvBlock(in_planes, init_planes, kernel_size=3, stride=1, padding=1)
                
        layers = []
        for ind, (dim_in, dim_out) in enumerate(in_out_dims):
            for _ in range(resnet_stacks):
                layers.append(ResnetBlock(dim_in, dim_in, groups=resnet_grnorm_groups))
            if ind in attention:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim))))
            
            # Use ConvBlock for downsampling with stride=2 for all but last layer
            if ind < len(in_out_dims) - 1:
                layers.append(ConvBlock(dim_in, dim_out, kernel_size=3, stride=2, padding=1))
            else:
                # No downsampling for the last layer
                layers.append(ConvBlock(dim_in, dim_out, kernel_size=3, stride=1, padding=1))
        
        self.layers = nn.Sequential(*layers)
        
        # Mid block configuration
        mid_layers = [MidBlock(
                            dim_out, 
                            dim_out, 
                            t_emb_dim=attn_dim, 
                            num_heads=attn_heads, 
                            num_layers=resnet_stacks, 
                            attn=True, 
                            norm_channels=resnet_grnorm_groups
                        ) for _ in range(resnet_stacks)
                    ]
        if not legacy_mid:
            mid_layers.insert(len(mid_layers) // 2, Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim))))
        self.mid_block = nn.Sequential(*mid_layers)
        
        self.post_enc = nn.Sequential(
            nn.GroupNorm(num_channels=dim_out, num_groups=resnet_grnorm_groups, eps=eps),
            nn.SiLU(),
            ConvBlock(dim_out, latent_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )
                
    def forward(self, x):
        x = self.init_conv(x)
        x = self.layers(x)
        x = self.mid_block(x)
        x = self.post_enc(x)
        return x