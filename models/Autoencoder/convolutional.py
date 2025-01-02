from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x