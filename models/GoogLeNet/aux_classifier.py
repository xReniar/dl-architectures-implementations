from torch import nn
import torch


class AuxClassifier(nn.Module):
    def __init__(self, in_features, conv_out, fc_out, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, conv_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, fc_out),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_out, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x:torch.Tensor):
        out: torch.Tensor = x
        out = self.avgpool(out)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out