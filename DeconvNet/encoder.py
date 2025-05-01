from torch import nn
import torch


def ConvBlock(in_channels:int, out_channels: int, n: int):
    layers = nn.Sequential()

    for x in range(0, n):
        layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels if x == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

    return layers

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 64, 2)
        self.conv2 = ConvBlock(64, 128, 2)
        self.conv3 = ConvBlock(128, 256, 3)
        self.conv4 = ConvBlock(256, 512, 3)
        self.conv5 = ConvBlock(512, 512, 3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 7),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        indices = {}
        for name, module in self.named_children():
            if name.startswith("conv"):
                x = module(x)
                x, index = self.max_pool(x)
                indices[name[-1]] = index
        
        x = self.bottleneck(x)

        return x, indices