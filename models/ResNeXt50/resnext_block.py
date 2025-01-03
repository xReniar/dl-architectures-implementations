from torch import nn
import torch


class ResNeXtBlock(nn.Module):
    def __init__(self, in_features:int, out_features:int, groups: int, cardinality:int, stride:int):
        super().__init__()

        intermediate_channels = int(out_features / 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(intermediate_channels, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

        if (stride != 1) or (in_features != out_features):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        else:
            self.downsample = None

    def forward(self, x:torch.Tensor):
        # residual path
        identity = x

        # normal path
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample != None:
            identity = self.downsample(identity)

        # final output
        x = self.relu(x + identity)
        
        return x