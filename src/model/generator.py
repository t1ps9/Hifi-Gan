import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        layers = []
        for dilation in dilations:
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Conv1d(channels, channels, kernel_size, stride=1,
                                    dilation=dilation, padding=(kernel_size * dilation - dilation) // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class MRF(nn.Module):
    def __init__(self, channels, kernels, dilations):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, k, d) for k, d in zip(kernels, dilations)
        ])

    def forward(self, x):
        return sum(block(x) for block in self.blocks) / len(self.blocks)


class Generator(nn.Module):
    def __init__(self, in_ch=80, out_ch=1, k_u=[16, 16, 4, 4],
                 k_r=[3, 7, 11], d_r=[[1, 1], [3, 1], [5, 1]], hid_u=512):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, hid_u, kernel_size=7, padding=3)

        self.up = nn.ModuleList()
        self.mrf_module = nn.ModuleList()
        current_channels = hid_u

        for kernel_size in k_u:
            stride = kernel_size // 2
            self.up.append(
                nn.ConvTranspose1d(
                    current_channels,
                    current_channels // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2
                )
            )
            self.mrf_module.append(
                MRF(current_channels // 2, k_r, d_r)
            )
            current_channels //= 2

        self.output_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(current_channels, out_ch, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.in_conv(x)
        for up_layer, mrf_layer in zip(self.up, self.mrf_module):
            x = nn.LeakyReLU(0.1)(x)
            x = up_layer(x)
            x = mrf_layer(x)
        return self.output_conv(x)
