from torch import nn
import torch


def NiN_block(in_features:int, out_features:int, kernel_size:int, stride:int = 1, padding:int = 0):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_features, out_features, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_features, out_features, kernel_size=1),
        nn.ReLU(inplace=True)
    )

class NiN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        # first block
        self.nin_block1 = NiN_block(in_features=3, out_features=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)

        # second block
        self.nin_block2 = NiN_block(in_features=96, out_features=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)

        # third block
        self.nin_block3 = NiN_block(in_features=256, out_features=384, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        # fourth block
        self.nin_block4 = NiN_block(in_features=384, out_features=num_classes, kernel_size=3, padding=1)
        
        # global average poooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = x
        out = self.nin_block1(out)
        out = self.pool1(out)
        out = self.nin_block2(out)
        out = self.pool2(out)
        out = self.nin_block3(out)
        out = self.pool3(out)
        out = self.nin_block4(out)
        out = self.gap(out)

        return out