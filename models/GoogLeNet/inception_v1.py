from torch import nn
import torch


class Inception_Block_v1(nn.Module):
    def __init__(self, in_features: int, branch_1:list, branch_2:list, branch_3:list, branch_4:list):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_features, branch_1[0], kernel_size=1),
            nn.BatchNorm2d(branch_1[0]),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_features, branch_2[0], kernel_size=1),
            nn.BatchNorm2d(branch_2[0]),
            nn.ReLU(),
            nn.Conv2d(branch_2[0], branch_2[1], kernel_size=3),
            nn.BatchNorm2d(branch_2[1]),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_features, branch_3[0], kernel_size=1),
            nn.BatchNorm2d(branch_3[0]),
            nn.ReLU(),
            nn.Conv2d(branch_3[0], branch_3[1], kernel_size=5,padding=2),
            nn.BatchNorm2d(branch_3[1]),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_features, branch_4[0], kernel_size=1),
            nn.BatchNorm2d(branch_4[0]),
            nn.ReLU()
        )


    def forward(self, x:torch.Tensor):
        b1_out = self.branch1(x)
        b2_out = self.branch2(x)
        b3_out = self.branch3(x)
        b4_out = self.branch4(x)

        return torch.cat([b1_out, b2_out, b3_out, b4_out])