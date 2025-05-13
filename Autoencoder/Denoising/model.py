import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_layer(3, 64, 2, 2)
        self.conv2 = self.conv_layer(64, 128, 2, 2)
        self.conv3 = self.conv_layer(128, 256, 2, 2)
        self.conv4 = self.conv_layer(256, 512, 3, 2)

        self.bottleneck = nn.Linear(86528, 2048)

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

    def forward(self, x: torch.Tensor):
        for name, module in self.named_children():
            if name.startswith("conv"):
                x = module(x)

        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(2048, 86528)
        self.deconv4 = self.deconv_layer(512, 256, 3, 2)
        self.deconv3 = self.deconv_layer(256, 128, 2, 2)
        self.deconv2 = self.deconv_layer(128, 64, 2, 2)
        self.deconv1 = self.deconv_layer(64, 3, 2, 2)

    def deconv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                output_padding=(1 if kernel_size == 3 else 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        x = self.bottleneck(x)
        x = x.view(-1, 512, 13, 13)
        
        for name, module in self.named_children():
            if name.startswith("deconv"):
                x = module(x)
        
        return x

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    

model = DAE()
image = torch.randn(1, 3, 224, 224)
image_noise = nn.Dropout(p=1.0)(image)
print(model(image).shape)