from torch import nn
import torch


class DenseLayer(nn.Module):
    def __init__(self,
        in_features: int,
        growth_rate: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=1, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1),
        )

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self,
        in_features: int,
        expansion: int, 
        growth_rate: int
    ) -> None:
        super().__init__()
        for i in range(0, expansion):
            self.add_module(str(i), DenseLayer(in_features, growth_rate))
            in_features += growth_rate

    def forward(self, x:torch.Tensor):
        processed = [x]
        for children in self.children():
            # get the output of the current dense layer
            x = children(x)

            # add it to the processed list
            processed.append(x)

            # concatenate by features 
            x = torch.cat(processed, dim=1)
        return x