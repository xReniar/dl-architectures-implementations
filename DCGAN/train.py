from discriminator import Discriminator
from generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import numpy as np


# device selection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


batch_size = 128
image_size = 64,
in_channels = 3
out_channels = 3
latent_vector_size = 100

generator_feature_maps_size = 64
discriminator_feature_maps_size = 64

dset = dataset.ImageFolder(
    root = "data/celeba",
    transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

dataloader = DataLoader(
    dataset=dset,
    batch_size=batch_size,
    shuffle=True
)

discriminator = Discriminator(in_channels, discriminator_feature_maps_size)
generator = Generator(out_channels, latent_vector_size, generator_feature_maps_size)


discriminator.apply(weights_init)
generator.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, latent_vector_size, 1, 1)

lr = 0.0002
beta1 = 0.5
num_epochs = 5

optimzerD = optim.Adam(discriminator.parameters(), lr = lr, betas = (beta1, 0.999))
optimzerG = optim.Adam(generator.parameters(), lr = lr, betas = (beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        pass