from torch import nn
import torch


class Inception_Block_v1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU()
        )


    def forward(self, x:torch.Tensor):
        pass