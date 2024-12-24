from residual_block import ResidualBlock
from torch import nn
import torch
    

class Resnet(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # resnet18
        self.conv2_x = self._make_layer(64, 2, [(64, 3), (64, 3)], stride=1)
        self.conv3_x = self._make_layer(64, 2, [(128, 3), (128, 3)], stride=2)
        self.conv4_x = self._make_layer(128, 2, [(256, 3), (256, 3)], stride=2)
        self.conv5_x = self._make_layer(256, 2, [(512, 3), (512, 3)], stride=2)

        # resnet34
        #self.conv2_x = self._make_layer(64, 3, [(64, 3), (64, 3)], stride=1)
        #self.conv3_x = self._make_layer(64, 4, [(128, 3), (128, 3)], stride=2)
        #self.conv4_x = self._make_layer(128, 6, [(256, 3), (256, 3)], stride=2)
        #self.conv5_x = self._make_layer(256, 3, [(512, 3), (512, 3)], stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def _make_layer(self, in_features:int, expansion:int, building_blocks:list, stride:int) -> nn.Sequential:
        layers = []

        feature_map_list = self._create_in_feature_maps_list(in_features, expansion, building_blocks)
        residual_block_list = self._build_residual_blocks(expansion, building_blocks, stride)

        for residual_block, feature_map in zip(residual_block_list, feature_map_list):
            layers.append(ResidualBlock(residual_block, feature_map))

        return nn.Sequential(*layers)

    def _build_residual_blocks(self, expansion:int, building_blocks:list, stride:int) -> list:
        group_blocks = list(map(lambda x: [x[0], x[1], 1, 1], building_blocks * expansion))
        group_blocks[0][2] = stride

        group_size = len(building_blocks)
        group_blocks = [group_blocks[i:i+group_size] for i in range(0, len(group_blocks), group_size)]
        return group_blocks
    
    # problems with resnet50 and greater architectures
    def _create_in_feature_maps_list(self, in_features, expansion, building_blocks) -> list:
        feature_maps:list = list(map(lambda x: x[0], building_blocks)) * expansion
        feature_maps.pop()
        feature_maps.insert(0, in_features)
        feature_maps = [feature_maps[i:i+len(building_blocks)] for i in range(0, len(feature_maps), len(building_blocks))]
        
        return feature_maps

    def forward(self, x):
        out:torch.Tensor = x
        out = self.conv1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out