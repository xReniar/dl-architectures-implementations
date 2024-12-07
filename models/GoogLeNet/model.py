from inception_v1 import Inception_Block_v1
from inception_v2 import Inception_Block_v2
from inception_v3 import Inception_Block_v3
from inception_v4 import Inception_Block_v4
from torch import nn
import torch


def auxiliary_classifier():
    pass

class GoogLeNet(nn.Module):
    def __init__(self, num_classes:int, inception_block_version:str):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7,stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.inception_1 = Inception_Block_v1()

    def forward(self, x:torch.Tensor):
        pass