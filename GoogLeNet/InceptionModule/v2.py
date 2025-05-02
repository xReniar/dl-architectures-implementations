from torch import nn
import torch


class InceptionModulev2(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # define features values for every branch
        self.in_features = args[0]
        self._1x1 = args[1]
        self._3x3_reduce_branch2 = args[2]
        self._3x3_branch2 = args[3]
        self._3x3_reduce_branch3 = args[4]
        self._3x3_1branch3 = args[5]
        self._3x3_2branch3 = args[6]
        self.pool_proj = args[7]

        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_features, self._1x1, kernel_size=1),
            nn.BatchNorm2d(self._1x1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_features, self._3x3_reduce_branch2, kernel_size=1),
            nn.BatchNorm2d(self._3x3_reduce_branch2),
            nn.ReLU(),
            nn.Conv2d(self._3x3_reduce_branch2, self._3x3_branch2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.in_features, self._3x3_reduce_branch3, kernel_size=1),
            nn.BatchNorm2d(self._3x3_reduce_branch3),
            nn.ReLU(),
            nn.Conv2d(self._3x3_reduce_branch3, self._3x3_1branch3, kernel_size=3,padding=2),
            nn.BatchNorm2d(self._3x3_1branch3),
            nn.ReLU(),
            nn.Conv2d(self._3x3_1branch3, self._3x3_2branch3, kernel_size=3,padding=2),
            nn.BatchNorm2d(self._3x3_2branch3),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.in_features, self.pool_proj, kernel_size=1),
            nn.BatchNorm2d(self.pool_proj),
            nn.ReLU()
        )
        self.aux_classifier = args[8]
        self.aux_value = None

    def forward(self, x:torch.Tensor, mode:bool = False):
        b1_out = self.branch1(x)
        b2_out = self.branch2(x)
        b3_out = self.branch3(x)
        b4_out = self.branch4(x)
        
        if self.aux_classifier != None and mode:
            self.aux_value = self.aux_classifier(x)

        return torch.cat([b1_out, b2_out, b3_out, b4_out], dim=1)