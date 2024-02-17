import math
import torch.nn as nn
import torch.nn.functional as F


class SpeechEncoder(nn.Module):
    def __init__(self, config, init_weights=True):
        super(SpeechEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(128),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(128),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=4, stride=1),
            nn.AvgPool2d((6,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.fc_net = nn.Sequential(
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU()
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1)
        out = self.fc_net(out)
        return out

    def _initialize_weights(self):
         for module in self.modules():
             if isinstance(module, nn.Conv2d):
                num_weight = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2./num_weight))
             elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
             elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

