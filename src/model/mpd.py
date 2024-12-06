import torch
import torch.nn as nn
import torch.nn.functional as F


class MPDSubDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))
        ])

    def forward(self, x):
        if x.shape[-1] % self.period != 0:
            pad_size = self.period - (x.shape[-1] % self.period)
            x = nn.functional.pad(x, (0, pad_size), mode='reflect')
        b, c, t = x.shape
        x = x.view(b, c, t // self.period, self.period)

        features = []
        for layer in self.conv_layers:
            x = layer(x)
            features.append(x)

        return x.view(b, -1), features


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([MPDSubDiscriminator(p) for p in periods])

    def forward(self, x):
        outputs = []
        features = []
        for discriminator in self.sub_discriminators:
            out, feats = discriminator(x)
            outputs.append(out)
            features.append(feats)
        return outputs, features
