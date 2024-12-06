import torch.nn as nn


class MSDSubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 128, kernel_size=41, stride=4, groups=4, padding=20),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 512, kernel_size=41, stride=4, groups=16, padding=20),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=64, padding=20),
            nn.LeakyReLU(0.1),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x):
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            features.append(x)
        return x.view(x.size(0), -1), features


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling_layers = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])
        self.sub_discriminators = nn.ModuleList([MSDSubDiscriminator() for _ in range(3)])

    def forward(self, x):
        outputs = []
        features = []
        for pool, discriminator in zip(self.pooling_layers, self.sub_discriminators):
            pooled_x = pool(x)
            out, feats = discriminator(pooled_x)
            outputs.append(out)
            features.append(feats)
        return outputs, features
