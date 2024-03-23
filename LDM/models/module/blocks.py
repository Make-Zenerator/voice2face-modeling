import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel=3,
                 stride=2,
                 padding=1):
        '''
        1D CNN block
        '''
        nn.Module.__init__(self)
        self.layers = []
        self.layers.append(
            nn.Conv1d(channel_in,
                      channel_out,
                      kernel,
                      stride,
                      padding,
                      bias=False))
        self.layers.append(
            nn.BatchNorm1d(channel_out, affine=True))
        self.layers.append(
            nn.ReLU(inplace=True))
        self.model = nn.Sequential(*self.layers)

    def forward(self, seq_in):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        seq_out = self.model(seq_in)
        return seq_out

class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, stride=1, padding=1, groups=4, dim_head=32):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(groups, dim_head)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None: # 추가 필요
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

class PreNorm(nn.Module):
    def __init__(self, fn, dim=32):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)     

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=None, groups=4):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, groups=groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=groups)
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.res_conv = nn.Identity()
        
        if t_emb_dim is not None:
            self.t_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        else:
            self.t_emb_layer = None

    def forward(self, x, t_emb=None):
        residual = x
        out = self.block1(x)
        if self.t_emb_layer is not None and t_emb is not None:
            out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.block2(out)
        out = out + self.res_conv(residual)
        return out

class DownBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 t_emb_dim, 
                 down_sample, 
                 num_heads, 
                 num_layers, 
                 norm_channels,
                 attn,  
                 cross_attn=False, 
                 context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(in_channels, out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
            ])
        
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.context_proj = nn.ModuleList([
                nn.Linear(context_dim, out_channels) for _ in range(num_layers)
            ])
            self.cross_attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
            ])
        
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x, context=None, t_emb=None):
        out = x
        for i in range(self.num_layers):
            out = self.resnet_blocks[i](out, t_emb)
            
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out
    
class MidBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 t_emb_dim, 
                 num_heads, 
                 num_layers, 
                 attn,
                 norm_channels, 
                 cross_attn=None, 
                 context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(in_channels, out_channels, t_emb_dim) for _ in range(num_layers + 1)
        ])
        
        if attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
            ])
        
        if cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.cross_attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
            ])
            self.context_proj = nn.ModuleList([
                nn.Linear(context_dim, out_channels) for _ in range(num_layers)
            ])
        
        # self.residual_input_conv = nn.ModuleList([
        #     nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
        #     for i in range(num_layers + 1)
        # ])

    def forward(self, x, context=None, t_emb=None):
        out = x
        for i in range(self.num_layers + 1):
            out = self.resnet_blocks[i](out, t_emb)
            
            if i < self.num_layers:
                if self.attn:
                    batch_size, channels, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, h * w)
                    in_attn = self.attention_norms[i](in_attn)
                    in_attn = in_attn.transpose(1, 2)
                    out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                    out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                    out = out + out_attn
                
                if self.cross_attn:
                    assert context is not None, "context cannot be None if cross attention layers are used"
                    batch_size, channels, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, h * w)
                    in_attn = self.cross_attention_norms[i](in_attn)
                    in_attn = in_attn.transpose(1, 2)
                    assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                    context_proj = self.context_proj[i](context)
                    out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                    out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                    out = out + out_attn
                
        return out

class UpBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 t_emb_dim, 
                 up_sample, 
                 num_heads, 
                 num_layers,
                 norm_channels,
                 attn, 
                 cross_attn=False,
                 context_dim=None
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(in_channels, out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
            ])
        
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.cross_attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
            ])
            self.context_proj = nn.ModuleList([
                nn.Linear(context_dim, out_channels) for _ in range(num_layers)
            ])
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if up_sample else nn.Identity()

    def forward(self, x, context=None, t_emb=None):
        x = self.up_sample_conv(x)
                
        out = x
        for i in range(self.num_layers):
            out = self.resnet_blocks[i](out, t_emb)
            
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
                        
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return out


class Inception1DBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_k2,
                 channel_k3,
                 channel_k5,
                 channel_k7):
        '''
        Basic building block of 1D Inception Encoder
        '''
        nn.Module.__init__(self)
        self.conv_k2 = None
        self.conv_k3 = None
        self.conv_k5 = None

        if channel_k2 > 0:
            self.conv_k2 = CNN1DBlock(
                channel_in,
                channel_k2,
                2, 2, 1)
        if channel_k3 > 0:
            self.conv_k3 = CNN1DBlock(
                channel_in,
                channel_k3,
                3, 2, 1)
        if channel_k5 > 0:
            self.conv_k5 = CNN1DBlock(
                channel_in,
                channel_k3,
                5, 2, 2)
        if channel_k7 > 0:
            self.conv_k7 = CNN1DBlock(
                channel_in,
                channel_k3,
                7, 2, 3)

    def forward(self, input):
        output = []
        if self.conv_k2 is not None:
            c2_out = self.conv_k2(input)
            output.append(c2_out)
        if self.conv_k3 is not None:
            c3_out = self.conv_k3(input)
            output.append(c3_out)
        if self.conv_k5 is not None:
            c5_out = self.conv_k5(input)
            output.append(c5_out)
        if self.conv_k7 is not None:
            c7_out = self.conv_k7(input)
            output.append(c7_out)
        #print(c2_out.shape, c3_out.shape, c5_out.shape, c7_out.shape)
        if output[0].shape[-1] > output[1].shape[-1]:
            output[0] = output[0][:, :, 0:-1]
        output = torch.cat(output, 1)
        #print(output.shape)
        return output
    