from torch import nn
import torch


class InceptionModulev1(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # define features values for every branch
        self.in_features:int = args[0]
        self._1x1:int = args[1]
        self._3x3_reduce:int = args[2]
        self._3x3:int = args[3]
        self._5x5_reduce:int = args[4]
        self._5x5:int = args[5]
        self.pool_proj:int = args[6]

        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_features, self._1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(self._1x1),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.in_features, self._3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(self._3x3_reduce),
            nn.ReLU(True),
            nn.Conv2d(self._3x3_reduce, self._3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self._3x3),
            nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.in_features, self._5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(self._5x5_reduce),
            nn.ReLU(True),
            nn.Conv2d(self._5x5_reduce, self._5x5, kernel_size=5,padding=2, bias=False),
            nn.BatchNorm2d(self._5x5),
            nn.ReLU(True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.in_features, self.pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.pool_proj),
            nn.ReLU(True)
        )
        self.aux_classifier:nn.Module = args[7]

    def forward(self, x:torch.Tensor, mode:bool = False):
        b1_out = self.branch1(x)
        b2_out = self.branch2(x)
        b3_out = self.branch3(x)
        b4_out = self.branch4(x)
        
        classifier = None
        if self.aux_classifier != None and mode:
            classifier = self.aux_classifier(x)

        return (torch.cat([b1_out, b2_out, b3_out, b4_out], dim=1), classifier)