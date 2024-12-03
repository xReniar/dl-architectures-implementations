from torch import nn
import torch


def NiN_block():
    return nn.Sequential(
        nn.LazyConv2d()
    )

class NiN(nn.Module):
    def __init__(self) -> None:
        super(NiN, self).__init__()
