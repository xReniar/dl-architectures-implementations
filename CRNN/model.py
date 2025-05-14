import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            self.block(1, 64, 3, 1, 1),
            self.block(64, 128, 3, 1, 1),
            self.block(128, 256, 3, 1, 1),
            self.block(256, 512, 3, 1, 1),
            self.block(512, 512, 3, 1, 1)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x:torch.Tensor):
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(3))

        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1, 512)

        return x
