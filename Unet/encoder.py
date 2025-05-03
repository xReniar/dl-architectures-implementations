import torch
import torch.nn as nn


def ConvBlocks(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 0,
    stride: int = 1
) -> nn.Sequential:
    layers = nn.Sequential()
    for i in range(0, 2):
        layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
    return layers

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlocks(1, 64)
        self.conv2 = ConvBlocks(64, 128)
        self.conv3 = ConvBlocks(128, 256)
        self.conv4 = ConvBlocks(256, 512)
        self.bottleneck = ConvBlocks(512, 1024)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        skip_values = {}
        for (name, module) in self.named_children():
            if name.startswith("conv"):
                x = module(x)
                skip_values[int(name[-1])] = x
                x = self.max_pool(x)

        x = self.bottleneck(x)

        return x, skip_values