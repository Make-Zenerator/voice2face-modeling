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
    

class SF2FEncoder(nn.Module):
    def __init__(self,model_config):
        super(SF2FEncoder, self).__init__()
        self.input_channel = model_config['input_channel']
        self.channels = model_config['channels']
        self.output_channel = model_config['output_channel']
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
        self.output_channel = self.output_channel

    def forward(self, x):
        if len(x.shape) == 4:
            B, N, C, L = x.shape
            x = x.view(B*N, C, L)
        else:
            B, C, L = x.shape
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



if __name__ == '__main__':
    import time
    # Demo Input
    log_mels = torch.ones((16, 40, 151))
    pos_log_mel = torch.ones((16, 48, 150)).transpose(1, 2)

    # Test Inception1DBlock
    incept_block = Inception1DBlock(
        channel_in=40,
        channel_k2=64,
        channel_k3=64,
        channel_k5=64,
        channel_k7=64)
    incept_block(log_mels)

    # Test V2F 1D CNN
    log_mel_segs = torch.ones((16, 20, 40, 150))
    v2f_id_cnn_kwargs = {
        'input_channel': 40,
        'channels': [256, 384, 576, 864],
        'output_channel': 512,
        'add_noise': False,
        'normalize': True,
        'return_seq': False,
        'inception_mode': True,
        }
    
    v2f_id_cnn_fuse = SF2FEncoder(v2f_id_cnn_kwargs)
    print(v2f_id_cnn_fuse)
    # print('SF2FEncoder Output shape:', v2f_id_cnn_fuse(log_mel_segs).shape)
    # v2f_id_cnn_fuse.print_param()
    # v2f_id_cnn_fuse.print_trainable_param()