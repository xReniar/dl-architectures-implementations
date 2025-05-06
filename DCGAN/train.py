from discriminator import Discriminator
from generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils


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
    root = "../data/celeba",
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

optimizerD = optim.Adam(discriminator.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr = lr, betas = (beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)

        output = discriminator(real_cpu).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, latent_vector_size, 1, 1, device=device)

        fake = generator(noise)
        label.fill(0)

        output = discriminator(fake.detach()).view(-1)

        errD_fake = criterion(output, label)

        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()

        generator.zero_grad()
        label.fill_(1)

        output = discriminator(fake).view(-1)

        errG = criterion(output, label)

        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1