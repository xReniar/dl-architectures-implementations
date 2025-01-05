from torch import nn
import torch


class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features, growth_rate):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, 4 * growth_rate, kernel_size=1, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1),
        )

    def forward(self, x:torch.Tensor):
        pass


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, expansion):
        super().__init__()
        self.dense_layers = []

        for _ in range(0, expansion):
            self.dense_layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(),
                    nn.BatchNorm2d(),
                    nn.ReLU(inplace=True),
                    nn.Conv2d()
                )
            )

    def forward(self, x:torch.Tensor):
        return x