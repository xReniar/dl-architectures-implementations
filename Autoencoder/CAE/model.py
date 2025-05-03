from encoder import Encoder
from decoder import Decoder
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_layer(1, 32, 2, 2)
        self.conv2 = self.conv_layer(32, 64, 2, 2)
        self.conv3 = self.conv_layer(64, 128, 3, 2)

        self.bottleneck = nn.Linear(1152, 10)

    def conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(10, 1152)
        self.deconv3 = self.deconv_layer(128, 64, 3, 2)
        self.deconv2 = self.deconv_layer(64, 32, 2, 2)
        self.deconv1 = self.deconv_layer(32, 1, 2, 2)

    def deconv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        x = self.bottleneck(x)
        x = x.view(-1, 128, 3, 3)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        
        return x

class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x