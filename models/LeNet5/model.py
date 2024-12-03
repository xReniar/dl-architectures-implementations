from torch import nn
import torch

class LeNet5(nn.Module):
    def __init__(self, num_classes) -> None:
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(84, num_classes)
        )

    def forward(self, x:torch.Tensor):
        out: torch.Tensor = x
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out