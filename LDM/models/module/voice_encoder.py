from torch import nn
import torch
import torch.nn.functional as F
from models.module.attention import Attention
from models.module.blocks import Inception1DBlock


class SF2FEncoder(nn.Module):
    def __init__(self,model_config):
        super(SF2FEncoder, self).__init__()
        self.self.input_channel = model_config['self.input_channel']
        self.self.channels = model_config['self.channels']
        self.self.output_channel = model_config['self.output_channel']
        self.add_noise=model_config['add_noise']
        self.normalize_embedding=model_config['normalize']
        self.return_seq=model_config['return_seq']
        self.inception_mode=model_config['inception_mode']

        if self.inception_mode:
            self.model = nn.Sequential(
                Inception1DBlock(
                    channel_in=self.input_channel,
                    channel_k2=self.channels[0]//4,
                    channel_k3=self.channels[0]//4,
                    channel_k5=self.channels[0]//4,
                    channel_k7=self.channels[0]//4),
                Inception1DBlock(
                    channel_in=self.channels[0],
                    channel_k2=self.channels[1]//4,
                    channel_k3=self.channels[1]//4,
                    channel_k5=self.channels[1]//4,
                    channel_k7=self.channels[1]//4),
                Inception1DBlock(
                    channel_in=self.channels[1],
                    channel_k2=self.channels[2]//4,
                    channel_k3=self.channels[2]//4,
                    channel_k5=self.channels[2]//4,
                    channel_k7=self.channels[2]//4),
                Inception1DBlock(
                    channel_in=self.channels[2],
                    channel_k2=self.channels[3]//4,
                    channel_k3=self.channels[3]//4,
                    channel_k5=self.channels[3]//4,
                    channel_k7=self.channels[3]//4),
                nn.Conv1d(self.channels[3], self.output_channel, 3, 2, 1, bias=True),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv1d(self.input_channel, self.channels[0], 3, 2, 1, bias=False),
                nn.BatchNorm1d(self.channels[0], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.channels[0], self.channels[1], 3, 2, 1, bias=False),
                nn.BatchNorm1d(self.channels[1], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.channels[1], self.channels[2], 3, 2, 1, bias=False),
                nn.BatchNorm1d(self.channels[2], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.channels[2], self.channels[3], 3, 2, 1, bias=False),
                nn.BatchNorm1d(self.channels[3], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.channels[3], self.output_channel, 3, 2, 1, bias=True),
            )
        self.add_noise = self.add_noise
        self.normalize_embedding = self.normalize_embedding
        self.return_seq = self.return_seq
        self.self.output_channel = self.output_channel

        # self.segments_fusion = segments_fusion
        # self.normalize_fusion = normalize_fusion
        # if segments_fusion:
        #     #self.attn_fuser = Attention(
        #     #    self.output_channel,
        #     #    self.output_channel,
        #     #    ignore_tanh=True)
        #     self.attn_fuser = \
        #         getattr(encoder_model_collection, fuser_arch)(**fuser_kwargs)

    def forward(self, x):
        # In case more than one mel segment per person is passed
        # if len(x.shape) == 4:
        #     fusion_mode = True
        #     B, N, C, L = x.shape
        #     #print('Fusion Mode On! Input Shape:', x.shape)
        #     x = x.view(B*N, C, L)
        # else:
        #     fusion_mode = False
        #     B, C, L = x.shape
        x = self.model(x)
        embeddings = F.avg_pool1d(x, x.size()[2], stride=1)
        embeddings = embeddings.view(embeddings.size()[0], -1, 1, 1)

        if self.normalize_embedding:
            embeddings = F.normalize(embeddings)
        if self.add_noise:
            noise = 0.05 * torch.randn(x.shape[0], x.shape[1], 1, 1)
            noise = noise.type(embeddings.type())
            embeddings = embeddings + noise
            if self.normalize_embedding:
                embeddings = F.normalize(embeddings)

        # # Restore Tensor shape
        # if fusion_mode:
        #     #print(embeddings.shape)
        #     C_emb = embeddings.shape[1]
        #     embeddings = embeddings.view(B, N, C_emb)
        #     # Attention fusion
        #     embeddings = self.attn_fuser(embeddings)
        #     if self.normalize_fusion:
        #         embeddings = F.normalize(embeddings)

        if self.return_seq:
            return embeddings, x
        else:
            return embeddings

    def print_param(self):
        print('All parameters:')
        for name, param in self.named_parameters():
            print(name)

    def print_trainable_param(model):
        print('Trainable Parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    # def train_fuser_only(self):
    #     print('Training Attention Fuser Only')
    #     for name, param in self.named_parameters():
    #         if 'attn_fuser' in name:
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False

    # def init_attn_fusion(self):
    #     self.attn_fuser = Attention(
    #         self.self.output_channel,
    #         self.self.output_channel,
    #         ignore_tanh=True)