from torch import nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x:torch.Tensor):
        pass
