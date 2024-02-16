import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config, init_weights=True):
        self.net = build_model(config)

        self.fc_net = nn.Sequential([
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096, biat=True),
            nn.ReLU()
        ])

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1)
        out = self.fc_net(out)
        return out

    def make_layers(cfg):
        layer = []
        input_channel = 2
        for v in cfg:
            if v == 'M':
                layer.append(nn.MaxPool2d(in_channel))
            elif:
                layer.append(nn.Conv2(input_channel, v, kenrel_size=4, stride=1))
                layer.append(nn.ReLU())
                layer.append(nn.BatchNorm2d(v))
                input_channel = v

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

