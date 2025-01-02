from torch import nn
import torch
import convolutional
import recurrent


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder:nn.Module = None
        self.decoder:nn.Module = None

    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x