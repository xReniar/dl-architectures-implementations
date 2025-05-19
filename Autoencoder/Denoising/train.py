import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import UNet
import matplotlib.pyplot as plt


# device selection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"

img_size = (224, 224)

train_dataset = datasets.STL10(
    root = "../../data",
    split="train",
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
    download = True
)

test_dataset = datasets.STL10(
    root = "../../data",
    split="test",
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
    download = True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False
)

model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
total_step = len(train_loader)

for epoch in range(epochs):
    # training
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        noise = torch.randn_like(images) * 0.3
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, i+1, total_step, loss.item()))

model.eval()
test_iter = iter(test_loader)
images, _ = next(test_iter)
images = images.to(device)
recon = model(images).detach().cpu()

# Show originals and reconstructed images
n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
    # Originals
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].cpu().permute(1, 2, 0))  # [C, H, W] → [H, W, C]
    plt.axis("off")
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon[i].permute(1, 2, 0))  # idem
    plt.axis("off")
plt.show()