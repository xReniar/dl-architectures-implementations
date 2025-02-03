from discriminator import Discriminator
from generator import Generator
from torch import nn
import torch


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x:torch.Tensor):
        pass