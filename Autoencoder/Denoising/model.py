import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.noise = nn.Dropout(0.5, True) 

    def forward(self, x: torch.Tensor):
        x = self.noise(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x