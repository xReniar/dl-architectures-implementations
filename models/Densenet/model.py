from densenet_block import DenseBlock
from torch import nn
import torch


class DenseNet(nn.Module):
    _growth_rate = 32

    def __init__(self, num_classes:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dense1 = DenseBlock()
        self.transition1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.dense2 = DenseBlock()
        self.transition2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.dense3 = DenseBlock()
        self.transition3 = nn.Sequential(
            nn.Conv2d(256 , 1, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.dense4 = DenseBlock()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(10, num_classes)

    def forward(self, x:torch.Tensor):
        x = self.conv(x)

        return x
    

def densenet_121(num_classes:int):
    pass

def densenet_169(num_classes:int):
    pass

def densenet_201(num_classes:int):
    pass

def densenet_264(num_classes:int):
    pass