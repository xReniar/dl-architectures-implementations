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
            nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.inception1 = Inception_Block_v1(192, [64], [96,128], [16,32], [32])
        self.inception2 = Inception_Block_v2()

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = x
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.inception1(out)
        #out = self.inception2()

        return out


x = torch.randn(1,3, 227,227)
net = GoogLeNet(10, inception_block_version="v1")
print(net(x).size())