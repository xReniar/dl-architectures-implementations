from torch import nn
import torch


class AuxClassifier(nn.Module):
    def __init__(self, in_features, conv_out, fc_out, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_features, conv_out, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(conv_out, fc_out)
        self.fc2 = nn.Linear(fc_out, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor):
        out: torch.Tensor = x
        out = self.avgpool(out)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out