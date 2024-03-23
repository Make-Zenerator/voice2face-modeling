import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.blocks import DownBlock, MidBlock, UpBlock
from models.module.encoder import Encoder
from models.module.decoder import Decoder


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        rng = max(1e-3, 1.0 / self.num_embeddings)
        self.embedding.weight.data.uniform_(-1*rng, rng)


    def forward(self, inputs):
        # Convert inputs from BxCxHxW to BxHxWxC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and reshape
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss for embedding
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        # Convert quantized from BxHxWxC back to BxCxHxW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss, quantized, encoding_indices
        
class VQVAE(nn.Module):
    def __init__(self, model_config):
        super(VQVAE, self).__init__()
        # Encoder Configuration
        self.encoder = Encoder(
                            in_planes=model_config['img_channels'],
                            init_planes=model_config['init_planes'],
                            planes_mults=model_config['planes_mults'],
                            resnet_grnorm_groups=model_config['resnet_groups'],
                            resnet_stacks=model_config['resnet_stacks'],
                            attention=model_config['attention'],
                            attn_heads=model_config['attn_heads'],
                            attn_dim=model_config['attn_dim'],
                            latent_dim=model_config['latent_dim'],
                            eps=model_config['eps'],
                            legacy_mid=model_config['legacy_mid']
                        )
        
        # Decoder Configuration
        self.decoder = Decoder(
                            in_planes=model_config['latent_dim'],
                            init_planes=model_config['init_planes'],
                            out_planes=model_config['img_channels'],
                            plains_divs=model_config['plains_divs'],
                            resnet_grnorm_groups=model_config['resnet_groups'],
                            resnet_stacks=model_config['resnet_stacks'],
                            up_mode=model_config['up_mode'],
                            scale=model_config['scale'],
                            attention=model_config['attention'],
                            attn_heads=model_config['attn_heads'],
                            attn_dim=model_config['attn_dim'],
                            eps=model_config['eps'],
                            legacy_mid=model_config['legacy_mid'],
                            tanh_out=model_config['tanh_out'],
                        )

        # Vector Quantizer
        self.x = VectorQuantizer(model_config['num_embeddings'], model_config['z_channels'], model_config['commitment_cost'])

        # Final layers for encoder and decoder
        self.encoder_final = nn.Conv2d(model_config['down_channels'][-1], model_config['z_channels'], kernel_size=1, padding=0)
        self.decoder_initial = nn.Conv2d(model_config['z_channels'], model_config['mid_channels'][-1], kernel_size=1, padding=0)
    
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = self.encoder_final(encoded)
        quantized, diff, _ = self.vq(encoded)
        return quantized, diff

    def decode(self, quantized):
        decoded_initial = self.decoder_initial(quantized)
        decoded = self.decoder(decoded_initial)
        return decoded

    def forward(self, x):
        quantized, diff = self.encode(x)
        decoded = self.decode(quantized)
        return decoded, quantized, diff