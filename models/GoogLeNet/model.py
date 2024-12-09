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
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception_3a = Inception_Block_v1(192, [64], [96,128], [16,32], [32])
        self.inception_3b = Inception_Block_v1(256, [128], [128,192], [32,96], [64])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4a = Inception_Block_v1(480, [192], [96, 208], [16,48], [64])
        self.inception_4b = Inception_Block_v1(512, [160], [112,224], [24,64], [64])
        self.inception_4c = Inception_Block_v1(512, [128], [128,256], [24,64], [64])
        self.inception_4d = Inception_Block_v1(512, [112], [144,288], [32,64], [64])
        self.inception_4e = Inception_Block_v1(528, [256], [160,320], [32,128], [128])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_5a = Inception_Block_v1(832, [256], [160,320], [32, 128], [128])
        self.inception_5b = Inception_Block_v1(832, [384], [192,384], [48,128], [128])
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024,num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = x
        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.maxpool3(out)
        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.maxpool4(out)
        out = self.inception_5a(out)
        out = self.inception_5b(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


x = torch.randn(1,3, 224,224)
net = GoogLeNet(1000, inception_block_version="v1")
print(net)