from torch import nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, blocks:list, in_features_list:list):
        super().__init__()

        for i, (block, in_features) in enumerate(zip(blocks, in_features_list)):
            out_features, kernel_size, stride, padding = block
            self.add_module(f"conv{i + 1}", nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ))

        # info needed for residual path
        stride = blocks[0][2]
        in_features = in_features_list[0]
        out_features = blocks[len(blocks) - 1][0]

        # downsampling if needed
        if (stride != 1) or (in_features != out_features):
            print(in_features, out_features)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_features)
            )
        else: 
            self.downsample = None

    def forward(self, x):
        # conv layers path
        out:torch.Tensor = x
        for name, module in self.named_children():
            if name != "downsample":
                out = module(out)

        # residual path
        identity:torch.Tensor = x
        if self.downsample != None:
            identity = self.downsample(identity)
        
        # sum
        out += identity
        
        return out