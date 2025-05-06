from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = None
        self.rnn = None

    def forward(self, x:torch.Tensor):
        x = self.mlp(x)
        x = self.rnn(x)

        return x