from torch import nn
import torch


def ConvBlock(in_channels:int, out_channels: int, n: int):
    layers = nn.Sequential()

    for x in range(0, n):
        current_out_channels = out_channels if x == n - 1 else in_channels
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                current_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(current_out_channels),
            nn.ReLU(inplace=True)
        ))

    return layers

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 7),
            nn.ReLU(inplace=True)
        )

        self.conv5 = ConvBlock(512, 512, 3)
        self.conv4 = ConvBlock(512, 256, 3)
        self.conv3 = ConvBlock(256, 128, 3)
        self.conv2 = ConvBlock(128, 64, 2)
        self.conv1 = ConvBlock(64, 2, 2)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, indices: dict[str, torch.Tensor]):
        x = self.bottleneck(x)

        for name, module in self.named_children():
            if name.startswith("conv"):
                x = self.unpool(x, indices[name[-1]])
                x = module(x)

        return x