from resnext_block import ResNeXtBlock
from torch import nn
import torch


class ResNeXt50(nn.Module):
    def __init__(self, cardinality: int, num_classes: int) -> None :
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = self._make_layer(64, 256, 3, cardinality)
        self.conv3 = self._make_layer(256, 512, 4, cardinality)
        self.conv4 = self._make_layer(512, 1024, 6, cardinality)
        self.conv5 = self._make_layer(1024, 2048, 3, cardinality)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(2048, num_classes)

    def _make_layer(self, in_features:int, out_features:int, expansion:int, cardinality:int):
        layers = []
        in_current = in_features
        layers.append(ResNeXtBlock(in_current, out_features, 0, cardinality, 2))
        in_current = out_features
        for _ in range(0, expansion - 1):
            layers.append(ResNeXtBlock(in_current, out_features, 0, cardinality, 1))

        return nn.Sequential(*layers)


    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    
image = torch.randn(1, 3, 224, 224)
net = ResNeXt50(32, 10)

print(net)