from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import numpy as np

class DCGAN(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(DCGAN, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.dec5 = DECNR2d(1 * self.nch_in,  8 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=self.norm, relu=0.0, drop=[])
        self.dec4 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0, drop=[])
        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0, drop=[])
        self.dec1 = Deconv2d(1 * self.nch_ker, 1 * self.nch_out,kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, img, labels):

        x = torch.cat((img, labels), -1).unsqueeze(2).unsqueeze(2)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.sigmoid(x)

        return x


class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])

        self.dec4 = DECNR2d(1 * 8 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec3 = DECNR2d(2 * 4 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec2 = DECNR2d(2 * 2 * self.nch_ker, 1 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec1 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_out, stride=2, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        x = torch.tanh(dec1)

        return x
   
class ConditionalUNetGAN(nn.Module):
    def __init__(self, noise_dim=100, nch_in=3, nch_out=3, condition_dim=9, nch_ker=64, norm='bnorm'):
        super(ConditionalUNetGAN, self).__init__()

        self.noise_dim = noise_dim
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.condition_dim = condition_dim

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.fc1 = nn.Sequential(
            nn.Linear(self.noise_dim+self.condition_dim,  2 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * self.nch_ker,  4 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4 * self.nch_ker,  16 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(16 * self.nch_ker,  32 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(32 * self.nch_ker,  self.nch_ker * self.nch_ker),
            nn.ReLU(0.2),
        )
        self.fc6 = nn.Linear(self.nch_ker * self.nch_ker,  3 * self.nch_ker * self.nch_ker)

        self.enc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])

        self.upc4 = DECNR2d(1 * 8 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.upc3 = DECNR2d(2 * 4 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.upc2 = DECNR2d(2 * 2 * self.nch_ker, 1 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.upc1 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_out, stride=2, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, noise, labels):
        batch_size = noise.size(0)

        x = torch.cat((noise, labels), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = x.reshape(batch_size, 3, 64, 64)

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        upc4 = self.upc4(enc4)
        upc3 = self.upc3(torch.cat([enc3, upc4], dim=1))
        upc2 = self.upc2(torch.cat([enc2, upc3], dim=1))
        upc1 = self.upc1(torch.cat([enc1, upc2], dim=1))

        x = torch.tanh(upc1)

        return x


class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)

        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]

            self.res = nn.Sequential(*res)

        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        if self.nblk:
            x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64    64 32
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128    32 16
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256     16 8
        # dsc4 : 32 x 32 x 256 -> 32 x 32 x 512     8  8
        # dsc5 : 32 x 32 x 512 -> 32 x 32 x 1       8  8

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


class ConditionMLPGAN(nn.Module):
    def __init__(self, nch_in, nch_out=3, nch_ker=64, norm='bnorm'):
        super(ConditionMLPGAN, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm
        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64    64 32
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128    32 16
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256     16 8
        # dsc4 : 32 x 32 x 256 -> 32 x 32 x 512     8  8
        # dsc5 : 32 x 32 x 512 -> 32 x 32 x 1       8  8

        self.fc1 = nn.Sequential(
            nn.Linear(1 * self.nch_in,  2 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * self.nch_ker,  4 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4 * self.nch_ker,  16 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(16 * self.nch_ker,  32 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(32 * self.nch_ker,  self.nch_ker * self.nch_ker),
            nn.ReLU(0.2),
        )
        self.fc6 = nn.Linear(self.nch_ker * self.nch_ker,  3 * self.nch_ker * self.nch_ker)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, img, labels):
        x = torch.cat((img, labels), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = torch.tanh(x)

        x = x.reshape(-1, 3, 64, 64)

        return x


class ConditionConvGenerator(nn.Module):
    def __init__(self, nch_in=100, condition_dim=9, nch_out=3, nch_ker=64, norm='bnorm'):
        super(ConditionConvGenerator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.nch_out = nch_out
        self.norm = norm
        self.condition_dim = condition_dim

        self.dec1 = DECNR2d(self.nch_in+self.condition_dim,  8 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=self.norm, relu=0.0, drop=[])
        self.dec2 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0, drop=[])
        self.dec4 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0, drop=[])
        self.dec5 = Deconv2d(1 * self.nch_ker, 1 * self.nch_out,kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, img, labels):
        x = torch.cat((img, labels), -1).unsqueeze(2).unsqueeze(2)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        x = torch.tanh(x)

        return x



class ConditionEmbeddingGAN(nn.Module):
    def __init__(self, nch_in, condition_dim=9, nch_ker=64):
        super(ConditionEmbeddingGAN, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.condition_dim = condition_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.nch_in+self.condition_dim,  2 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * self.nch_ker,  4 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4 * self.nch_ker,  16 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(16 * self.nch_ker,  32 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(32 * self.nch_ker,  self.nch_ker * self.nch_ker),
            nn.ReLU(0.2),
        )
        self.fc6 = nn.Linear(self.nch_ker * self.nch_ker,  3 * self.nch_ker * self.nch_ker)

    def forward(self, noise, labels):
        batch_size = noise.size(0)

        x = torch.cat((noise, labels), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = torch.tanh(x)

        x = x.reshape(batch_size, 3, 64, 64)

        return x

class ConditioEmbeddingDiscriminator(nn.Module):
    def __init__(self, input_size=(3,64,64), nch_ker=64, condition_dim=9):
        super(ConditioEmbeddingDiscriminator, self).__init__()

        self.input_size = input_size # (3,64,64)
        self.nch_ker = nch_ker
        self.condition_dim = condition_dim

        self.fc1 = nn.Sequential(
            nn.Linear(64*64*3+self.condition_dim, 8 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(8 * self.nch_ker,  4 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4 * self.nch_ker,  2 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(2 * self.nch_ker,  1 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(1 * self.nch_ker,  1),
        )

    def forward(self, img, labels):
        batch_size = img.size(0)
        x = img.flatten(start_dim=1)
        x = torch.cat((x, labels),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        x = torch.sigmoid(x)
        
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


## finetuning gan
class ConditionEmbeddingFrozenGAN(nn.Module):
    def __init__(self, nch_in, nch_out=3, nch_ker=64, norm='bnorm'):
        super(ConditionEmbeddingFrozenGAN, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        self.fc1 = nn.Sequential(
            nn.Linear(1 * self.nch_in,  2 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * self.nch_ker,  4 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4 * self.nch_ker,  16 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(16 * self.nch_ker,  32 * self.nch_ker),
            nn.ReLU(0.2),
            nn.Dropout(0.1)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(32 * self.nch_ker,  self.nch_ker * self.nch_ker),
            nn.ReLU(0.2),
        )
        self.fc6 = nn.Linear(self.nch_ker * self.nch_ker,  3 * self.nch_ker * self.nch_ker)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = torch.tanh(x)

        x = x.reshape(-1, 3, 64, 64)

        return x
