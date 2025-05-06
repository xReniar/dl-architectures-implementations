from torch import nn
import torch


def vgg_block(in_channels: int, out_channels: int, num_of_blocks: int):
    block = nn.Sequential()
    for i in range(num_of_blocks):
        block.append(nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=3,stride=1,padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ))

    return block

class vgg16_bn(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv1 = vgg_block(3, 64, 2)
        self.conv2 = vgg_block(64, 128, 2)
        self.conv3 = vgg_block(128, 256, 3)
        self.conv4 = vgg_block(256, 512, 3)
        self.conv5 = vgg_block(512, 512, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096,num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = x
        for (name, module) in self.named_children():
            if name.startswith("conv"):
                out = module(out)
                out = self.maxpool(out)
    
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out