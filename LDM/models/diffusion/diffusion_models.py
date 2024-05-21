import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict

from models.module.blocks import DownBlock, UpBlock, MidBlock
from models.module.attention import CrossAttention
from models.module.vqvae import VQVAE
from models.module.ema import EMA
from models.module.noise_scheduler import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from modules.losses import get_loss_function


def get_betas(timesteps: int, scheduler_name='linear'):
    """
    Computes betas according to the scheduler type and required parameters.
    The function has default parameters.
    SF
    """
    if 'cosine' in scheduler_name:
        scheduler = cosine_beta_schedule
        s = 0.008
        betas = scheduler(timesteps, s)
    elif 'linear' in scheduler_name:
        scheduler = linear_beta_schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = scheduler(timesteps, beta_start, beta_end)
    elif 'quadratic' in scheduler_name:
        scheduler = quadratic_beta_schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = scheduler(timesteps, beta_start, beta_end)
    elif 'sigmoid' in scheduler_name:
        scheduler = sigmoid_beta_schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = scheduler(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f'Scheduler type of "{scheduler_name}" is not recognized')
    return torch.tensor(betas)

def extract(a, t, x_shape):
    # Taken from Annotated diffusion model by HuggingFace
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_params(betas, timesteps):
    # Adopted from Annotated diffusion model by HuggingFace
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return (betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(t.device), beta.to(t.device)], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(model_args, seq, model, b, eta):
    with torch.no_grad():
        x = model_args[0]
        self_cond = model_args[1]
        clas_lbls = model_args[2]
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        progress_bar = tqdm(zip(reversed(seq), reversed(seq_next)),
                            desc=f'DDIM Sampling', total=len(seq),
                            mininterval=0.5, leave=False,
                            disable=False, colour='#F39C12',
                            dynamic_ncols=True)

        #for i, j in zip(reversed(seq), reversed(seq_next)):
        for i,j in progress_bar:
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t, x_self_cond=self_cond, lbls=clas_lbls).detach()
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.detach().to('cpu'))

    return xs, x0_preds

def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.detach().to('cpu'))
    return xs, x0_preds

class UNetWithCrossAttention(nn.Module):
    def __init__(self, model_config):
        super(UNetWithCrossAttention, self).__init__()
        self.image_channels = model_config['image_channels']
        self.model_channels = model_config['model_channels']
        self.num_res_blocks = model_config['num_res_blocks']
        self.num_heads = model_config['num_heads'] 
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.init_conv = nn.Conv2d(self.image_channels, self.model_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList([
                                DownBlock(
                                    in_channels=self.model_channels * (2 ** i), 
                                    out_channels=self.model_channels * (2 ** (i + 1)), 
                                    t_emb_dim=self.dim_head, 
                                    down_sample=True, 
                                    num_heads=self.num_heads, 
                                    num_layers=2, 
                                    norm_channels=1,
                                    attn=False, 
                                    cross_attn=True,
                                    context_dim=self.model_channels * (2 ** (i + 1))
                                    )
                                for i in range(self.num_res_blocks)
                            ])
        
        self.mid_blocks = MidBlock(in_channels=self.model_channels * (2 ** len(self.num_res_blocks)),
                                   out_channels=self.model_channels * (2 ** len(self.num_res_blocks)),
                                   t_emb_dim=self.dim_head,
                                   num_heads=self.num_heads,
                                   num_layers=1,
                                   norm_channels=1,
                                   attn=False,
                                   ) 

        self.up_blocks = nn.ModuleList([
                                UpBlock(self.model_channels * (2 ** (i + 1)),
                                        self.model_channels * (2 ** i), 
                                        t_emb_dim=self.dim_head, 
                                        down_sample=True, 
                                        num_heads=self.num_heads, 
                                        num_layers=2, 
                                        norm_channels=1,
                                        attn=False, 
                                        cross_attn=True,
                                        context_dim=self.model_channels * (2 ** i)
                                        )
                                    for i in reversed(range(self.num_res_blocks))
                                ])
        
        self.cross_attn_blocks = nn.ModuleList([
                                        CrossAttention(self.model_channels * (2 ** i), 
                                                    heads=self.num_heads, 
                                                    dim_head=self.dim_head, 
                                                    dropout=self.dropout)
                                        for i in range(self.num_res_blocks)
                                    ])
        
        self.final_conv = nn.Conv2d(self.model_channels, self.image_channels, 1)

        self.ema = EMA(self)

    def forward(self, x, context=None, t=None):
        x = self.init_conv(x)
        residuals = []
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x, context, t)
            residuals.append(x)

        x = self.mid_blocks(x, t)

        for i, (up_block, residual) in enumerate(zip(self.up_blocks, reversed(residuals))):
            x = F.interpolate(x, scale_factor=2, mode='nearest') 
            x = torch.cat([x, residual], dim=1)
            x = up_block(x, context, t)
            # if context is not None:
            #     x = self.cross_attn_blocks[i](x, context)

        x = self.final_conv(x)
        return x
    
    def update_ema(self):
        self.ema.update()

    def apply_ema(self):
        self.ema.apply_shadow()

    def restore_weights(self):
        self.ema.restore()

class LatentDiffusion(nn.Module):
    def __init__(self, unet, voice_encoder, vqvae, betas):
        super(LatentDiffusion, self).__init__()
        self.unet = unet
        self.voice_encoder = voice_encoder
        self.autoencoder = vqvae
        self.num_timesteps = len(betas)
        self.sqrt_alphas_cumprod = torch.sqrt(1.0 - torch.tensor(betas).cumprod(dim=0))
        # self.vector_quantizer = self.autoencoder.vq

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            x_start * self.sqrt_alphas_cumprod[t].view(1, -1, 1, 1)
            + noise * torch.sqrt(1.0 - self.sqrt_alphas_cumprod[t].view(1, -1, 1, 1))
        )
    
    def p_sample(self, size, x_self_cond=None,
                 classes=None, last = True,
                 eta: float = 1.0):
        """ Posterior sample """
        x = torch.randn(*size, device=self.dev)
        seq = range(0, self.timesteps, self.sample_every)
        seq = [int(s) for s in list(seq)]
        model_args = (x, x_self_cond, classes)
        xs = generalized_steps(model_args, seq, self.model, self.betas, eta=eta)
        if last:
            return xs[0][-1]
        else:
            return xs

    def forward(self, x, voice_condition, t):
        t_emb = self.get_time_embedding(x.size(0), t)
        # Vector quantization
        # _, diff, _ = self.vector_quantizer(x)
        q_loss, diff, _ = self.autoencoder.encode(x)

        # Apply diffusion process
        x_noisy = self.q_sample(diff, t_emb)

        # Encode voice condition
        voice_emb, _ = self.voice_encoder(voice_condition)

        # Generate latent representation using UNet
        latent = self.unet(x_noisy, context=voice_emb, t=t_emb)
        
        # Decode latent representation
        # decoded = self.unet.decode(latent)
        decoded = self.autoencoder.decode(latent)
        
        return decoded
    
    def get_time_embedding(time_steps, temb_dim):
        r"""
        Convert time steps tensor into an embedding using the
        sinusoidal time embedding formula
        :param time_steps: 1D tensor of length batch size
        :param temb_dim: Dimension of the embedding
        :return: BxD embedding representation of B time steps
        """
        assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
        
        # factor = 10000^(2i/d_model)
        factor = 10000 ** ((torch.arange(
            start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
        )
        
        # pos / factor
        # timesteps B -> B, 1 -> B, temb_dim
        t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb
    

## 참고
class Diffusion(nn.Module):
    def __init__(self, noise_dict: Dict, model,
                 timesteps: int = 500, loss:str = 'l2',
                 sample_every: int = 20,
                 device: str = 'cuda'):
        self.timesteps = timesteps
        self.sample_every = sample_every
        self.dev = device
        betas =  get_betas(timesteps=timesteps, noise_kw = noise_dict) # noise_dict : model_config['noise_dict']
        dif_params = forward_diffusion_params(betas, timesteps)
        self.betas = dif_params[0]
        self.sqrt_recip_alphas = dif_params[1]
        self.sqrt_alphas_cumprod = dif_params[2]
        self.sqrt_one_minus_alphas_cumprod = dif_params[3]
        self.posterior_variance = dif_params[4]
        self.loss = get_loss_function(loss)
        self.model = model

    def q_sample(self, x_start, t, noise=None):
        """
        Prior sample
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward_diffusion(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start,
                                t=t, noise=noise)
        return x_noisy

    def get_loss(self, x_start, t, noise=None, x_self_cond=None, classes=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start,
                                t=t, noise=noise)

        predicted_noise = self.model(x_noisy, t, x_self_cond=x_self_cond, lbls=classes)
        return self.loss(noise, predicted_noise)

    def p_sample(self, size, x_self_cond=None,
                 classes=None, last = True,
                 eta: float = 1.0):
        """ Posterior sample """
        x = torch.randn(*size, device=self.dev)
        seq = range(0, self.timesteps, self.sample_every)
        seq = [int(s) for s in list(seq)]
        model_args = (x, x_self_cond, classes)
        xs = generalized_steps(model_args, seq, self.model, self.betas, eta=eta)
        if last:
            return xs[0][-1]
        else:
            return xs
