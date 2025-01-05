from .densenet_block import DenseBlock
from torch import nn
import torch


class DenseNet(nn.Module):
    # only works with 32
    __growth_rate = 32

    def __init__(self, expansions:list, num_classes:int):
        super().__init__()
        self.__transition3_in = 256 + (expansions[2] * self.__growth_rate)
        self.__classifier_in = (self.__transition3_in // 2) + (expansions[3] * self.__growth_rate)


        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dense1 = DenseBlock(64, expansions[0], self.__growth_rate)
        self.transition1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.dense2 = DenseBlock(128, expansions[1], self.__growth_rate)
        self.transition2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.dense3 = DenseBlock(256, expansions[2], self.__growth_rate)
        self.transition3 = nn.Sequential(
            nn.Conv2d(self.__transition3_in, self.__transition3_in // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.__transition3_in // 2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.dense4 = DenseBlock(self.__transition3_in // 2, expansions[3], self.__growth_rate)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(self.__classifier_in, num_classes)

    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)
        x = self.dense4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    

def densenet_121(num_classes:int):
    expansions = [6, 12, 24, 16]
    return DenseNet(expansions, num_classes)

def densenet_169(num_classes:int):
    expansions = [6, 12, 32, 32]
    return DenseNet(expansions, num_classes)

def densenet_201(num_classes:int):
    expansions = [6, 12, 48, 32]
    return DenseNet(expansions, num_classes)

def densenet_264(num_classes:int):
    expansions = [6, 12, 64, 48]
    return DenseNet(expansions, num_classes)