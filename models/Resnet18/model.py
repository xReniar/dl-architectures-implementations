from torch import nn
import torch


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass