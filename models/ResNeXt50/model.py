from resnext_block import ResNeXtBlock
from torch import nn
import torch


class ResNeXt50(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(10,num_classes)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)

        return x
    


image = torch.randn(1, 3, 224, 224)
net = ResNeXt50(10)


print(net(image).size())