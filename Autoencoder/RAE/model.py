from encoder import Encoder
from decoder import Decoder
from torch import nn
import torch


class RAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x