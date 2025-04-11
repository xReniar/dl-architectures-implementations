from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_x = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_x = nn.Sequential()
        self.conv3_x = nn.Sequential()
        self.conv4_x = nn.Sequential()
        self.conv5_x = nn.Sequential()

    def forward(self, x:torch.Tensor):
        x = self.conv1_x(x)
        x = self.maxpool_1(x)
        #x = self.conv2_x(x)
        #x = self.conv3_x(x)
        #x = self.conv4_x(x)
        #x = self.conv5_x(x)

        return x
    

x = torch.randn(3,224,224)
model = Encoder()
y = model(x)
print(y.size())