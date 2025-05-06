from torch import nn
import torch


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = None
        self.mlp = None

    def forward(self, x:torch.Tensor):
        x = self.rnn(x)
        x = self.mlp(x)

        return x