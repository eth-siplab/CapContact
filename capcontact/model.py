import torch
from torch import nn
import math


class Generator(nn.Module):
    """
        Generator for 8x cross-modality super-resolution
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
            )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=64) for _ in range(5)],
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
            )

        self.upscaling_blocks = nn.Sequential(
            *[UpscalingBlock(64, upscale_factor=2) for _ in range(3)]
            )

        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, padding=4),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.residual_blocks(x) + x
        x = self.upscaling_blocks(x)
        return self.out_layer(x)


class UpscalingBlock(nn.Module):

    def __init__(self, channels, upscale_factor=2):
        super(UpscalingBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*(upscale_factor**2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
            )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        return x+self.net(x)


class Critic(nn.Module):
    """
        Critic / Discriminator network for Wasserstein GAN - GP
    """
    def __init__(self):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),

            VGGConvolutionalBlock(in_channels=32, out_channels=64),
            VGGConvolutionalBlock(in_channels=64, out_channels=128),
            VGGConvolutionalBlock(in_channels=128, out_channels=256),
            VGGConvolutionalBlock(in_channels=256, out_channels=512),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        return self.net(x).view(-1)


class VGGConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(VGGConvolutionalBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False),
            )

    def forward(self, x):
        return self.net(x)
