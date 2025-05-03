import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_vector_size: int,
        feature_maps_size: int,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, feature_maps_size * 8, 4, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size * 8),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps_size * 8, feature_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size * 4),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps_size * 4, feature_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size * 2),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps_size * 2, feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps_size, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for (_, module) in self.named_children():
            x = module(x)

        return x