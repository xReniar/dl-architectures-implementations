from InceptionModule.v1 import Inception_Block_v1
from InceptionModule.v2 import Inception_Block_v2
from InceptionModule.v3 import Inception_Block_v3
from InceptionModule.v4 import Inception_Block_v4
from aux_classifier import AuxClassifier
from torch import nn
import torch


class GoogLeNet(nn.Module):
    def __init__(self, num_classes:int, inception_block_version:str):
        super().__init__()
        self.inception_module_version = inception_block_version
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
        self.inception_3a = self.inception_module(192, 64, 96, 128, 16, 32, 32, None)
        self.inception_3b = self.inception_module(256, 128, 128, 192, 32, 96, 64, None)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4a = self.inception_module(480, 192, 96, 208, 16,48, 64, None)
        self.inception_4b = self.inception_module(512, 160, 112,224, 24,64, 64, None)
        self.inception_4c = self.inception_module(512, 128, 128,256, 24,64, 64, None)
        self.inception_4d = self.inception_module(512, 112, 144,288, 32,64, 64, None)
        self.inception_4e = self.inception_module(528, 256, 160,320, 32,128, 128, None)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_5a = self.inception_module(832, 256, 160,320, 32, 128, 128, None)
        self.inception_5b = self.inception_module(832, 384, 192,384, 48, 128, 128, None)
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Dropout(p=0.4, inplace=True)
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
    
    def inception_module(self, *args):
        blocks = {
            "v1": Inception_Block_v1(*args),
            "v2": Inception_Block_v2(),
            "v3": Inception_Block_v3(),
            "v4": Inception_Block_v4()
        }

        return blocks[self.inception_module_version]


x = torch.randn(1,3, 224,224)
net = GoogLeNet(10, inception_block_version="v1")
print(net(x))