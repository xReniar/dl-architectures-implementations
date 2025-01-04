from .resnext_block import ResNeXtBlock
from torch import nn
import torch


class ResNeXt50(nn.Module):
    def __init__(self, cardinality:int, groups_width:int, num_classes: int) -> None :
        super().__init__()
        self.cardinality = cardinality
        self.grous_width = groups_width

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = self._make_layer(64, 256, 3)
        self.conv3 = self._make_layer(256, 512, 4)
        self.conv4 = self._make_layer(512, 1024, 6)
        self.conv5 = self._make_layer(1024, 2048, 3)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(2048, num_classes)

    def _make_layer(self, in_features:int, out_features:int, expansion:int):
        layers = []
        in_current = in_features
        layers.append(ResNeXtBlock(in_current, out_features, self.cardinality, self.grous_width, 2))
        in_current = out_features
        for _ in range(0, expansion - 1):
            layers.append(ResNeXtBlock(in_current, out_features, self.cardinality, self.grous_width, 1))

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
    
'''
def visualize_model():
    from torchviz import make_dot
    
    model = ResNeXt50(32, 4, 10)
    inputs = torch.randn(1, 3, 224, 224)
    y = model(inputs)
    make_dot(y, params=dict(model.named_parameters())).render("model", format="png")
'''