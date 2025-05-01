from encoder import Encoder
from decoder import Decoder
from torch import nn
import torch


class DeconvNet(nn.Module):
    '''
    Input size: (1, 3, 224, 224)
    '''
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x:torch.Tensor):
        x, indices = self.encoder(x)
        x = self.decoder(x, indices)

        return x