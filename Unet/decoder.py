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
        current_out_channels = out_channels if i == 1 else in_channels
        layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=current_out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(current_out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                current_out_channels,
                current_out_channels,
                kernel_size=3
            )
        ))
    return layers

def copy_and_crop(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_y = source.size(2) - target.size(2)
    diff_x = source.size(3) - target.size(3)
    return source[:, :, diff_y // 2 : diff_y // 2 + target.size(2),
                        diff_x // 2 : diff_x // 2 + target.size(3)]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=2)
        self.conv4 = ConvBlocks(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=2)     
        self.conv3 = ConvBlocks(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=2)
        self.conv2 = ConvBlocks(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2)
        self.conv1 = ConvBlocks(128, 64)

        self.head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, skip_values: dict[int, torch.Tensor]):
        x = self.upconv4(x)
        cropped = copy_and_crop(skip_values[4], x)
        x = self.conv4(torch.cat([x, cropped], dim=1))

        x = self.upconv3(x)
        cropped = copy_and_crop(skip_values[3], x)
        x = self.conv3(torch.cat([x, cropped], dim=1))

        x = self.upconv2(x)
        cropped = copy_and_crop(skip_values[2], x)
        x = self.conv2(torch.cat([x, cropped], dim=1))

        x = self.upconv1(x)
        cropped = copy_and_crop(skip_values[1], x)
        x = self.conv1(torch.cat([x, cropped], dim=1))

        x = self.head(x)

        return x