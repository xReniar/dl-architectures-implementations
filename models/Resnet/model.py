from copy import deepcopy
from torch import nn
import torch


# fare la shortcut
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

    def forward(self, x):
        out:torch.Tensor = x
        for module in self.children():
            out = module(out)
        
        return out
    

class Resnet(nn.Module):
    def __init__(self, parameter_layers:int, num_classes:int):
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

        # resnet50
        #self.conv2_x = self._make_layer(64, 3, [(64, 1), (64, 3), (256, 1)], stride=1)
        #self.conv3_x = self._make_layer(256, 4, [(128, 1), (128, 3), (512, 1)], stride=2)
        #self.conv4_x = self._make_layer(512, 6, [(256, 1), (256, 3), (1024, 1)], stride=2)
        #self.conv5_x = self._make_layer(1024, 3, [(512, 1), (512, 3), (2048, 1)], stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
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
    
    # il problema è resnet50 in su non fa match
    def _create_in_feature_maps_list(self, in_features, expansion, building_blocks) -> list:
        '''
        #feature_maps = [block[0] for block in building_blocks] * expansion * len(building_blocks)
        feature_maps = [[deepcopy(block[0]),deepcopy(block[0])] for block in building_blocks] * expansion
        feature_maps[0][0] = in_features
        print(feature_maps)
        for i in range(1, len(feature_maps)):
            print(feature_maps[i])
        #feature_maps = [[feature_maps[i][0], feature_maps[i - 1][1]] for i in range(1, len(feature_maps))] # modificare questo
        #feature_maps[0][0] = in_features
        print(feature_maps)

        group_size = len(building_blocks)
        feature_maps = [feature_maps[i:i+2] for i in range(0, len(feature_maps), 2)]
        feature_maps = [feature_maps[i:i+group_size] for i in range(0, len(feature_maps), group_size)]
        '''
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

        return out


image = torch.randn(1, 3, 224, 224)
net = Resnet(parameter_layers=18, 
             num_classes=10)
print(net(image).size())
print(net)