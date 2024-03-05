import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from utils import GLU


class ConditionalGenrator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=9):
        super(ConditionalGenrator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim

        # Fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 512),
            nn.BatchNorm1d(512),
            GLU(),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            GLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
        )

        # Transpose convolution layers
        self.transpose_conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        # Concatenate noise and condition vectors
        x = torch.cat((noise, condition), dim=1)
        
        # Fully connected layer
        x = self.fc_layers(x)
        x = x.reshape(-1, 1, 32, 32)  # Reshape to image tensor shape
        
        # Transpose convolution layers
        x = self.transpose_conv(x)

        return x

    def load_model(self, checkpoint):
        model = torch.load_state_dict(checkpoint)
        return model


class Discriminator(nn.Module):
    def __init__(self, condition_dim=9) -> None:
        super().__init__()
        self.condition_dim = condition_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_layers(x)
        out = self.global_avg_pool(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out
    


class GenderDiscriminator(nn.Module):
    def __init__(self):
        super(GenderDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def load_model(self, checkpoint):
        model = torch.load_state_dict(checkpoint)
        return model

class AgeDiscriminator(nn.Module):
    def __init__(self):
        super(AgeDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(128, 8)

        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = self.fc_layer(x)
        x = F.sigmoid(x)
        return x

    def load_model(self, checkpoint):
        model = torch.load_state_dict(checkpoint)
        return model
