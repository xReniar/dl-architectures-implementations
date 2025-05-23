from InceptionModule.v1 import InceptionModulev1
from InceptionModule.v2 import InceptionModulev2
from InceptionModule.v3 import InceptionModulev3
from InceptionModule.v4 import InceptionModulev4
from aux_classifier import AuxClassifier
from torch import nn
import torch


class GoogLeNet(nn.Module):
    def __init__(self, num_classes:int, inception_block_version:str):
        super().__init__()
        self.inception_module_version = inception_block_version
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception_3a = self.inception_module(192, 64, 96, 128, 16, 32, 32, None)
        self.inception_3b = self.inception_module(256, 128, 128, 192, 32, 96, 64, None)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4a = self.inception_module(480, 192, 96, 208, 16, 48, 64, None)
        self.inception_4b = self.inception_module(512, 160, 112, 224, 24, 64, 64, AuxClassifier(512, 128, 1024, num_classes))
        self.inception_4c = self.inception_module(512, 128, 128, 256, 24, 64, 64, None)
        self.inception_4d = self.inception_module(512, 112, 144, 288, 32, 64, 64, None)
        self.inception_4e = self.inception_module(528, 256, 160, 320, 32, 128, 128, AuxClassifier(528, 128, 1024, num_classes))
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_5a = self.inception_module(832, 256, 160,320, 32, 128, 128, None)
        self.inception_5b = self.inception_module(832, 384, 192,384, 48, 128, 128, None)
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Dropout(p=0.4, inplace=True)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out:torch.Tensor = x
        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out, _ = self.inception_3a(out)
        out, _ = self.inception_3b(out)
        out = self.maxpool3(out)
        out, _ = self.inception_4a(out)
        out, aux1 = self.inception_4b(out, self.training)
        out, _ = self.inception_4c(out)
        out, _ = self.inception_4d(out)
        out, aux2 = self.inception_4e(out, self.training)
        out = self.maxpool4(out)
        out, _ = self.inception_5a(out)
        out, _ = self.inception_5b(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return aux1, aux2, out
    
    def inception_module(self, *args) -> InceptionModulev1 | InceptionModulev2 | InceptionModulev3 | InceptionModulev4:
        blocks = {
            "v1": InceptionModulev1(*args),
            #"v2": InceptionModulev2(*args),
            #"v3": InceptionModulev3(*args),
            #"v4": InceptionModulev4(*args)
        }

        return blocks[self.inception_module_version]