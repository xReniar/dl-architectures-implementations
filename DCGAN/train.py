from discriminator import Discriminator
from generator import Generator
from torch import nn
import torch



batch_size = 128
image_size = 64,
in_channels = 3
out_channels = 3
latent_vector_size = 100

generator_feature_maps_size = 64
discriminator_feature_maps_size = 64


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

discriminator = Discriminator(in_channels, discriminator_feature_maps_size)
generator = Generator(out_channels, latent_vector_size, generator_feature_maps_size)


discriminator.apply(weights_init)
generator.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, latent_vector_size, 1, 1)