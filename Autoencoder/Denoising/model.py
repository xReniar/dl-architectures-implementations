from encoder import Encoder
from decoder import Decoder
from torch import nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x:torch.Tensor):
        x, skip_values = self.encoder(x)
        x = self.decoder(x, skip_values)

        return x