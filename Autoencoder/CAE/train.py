import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from model import CAE
import matplotlib.pyplot as plt


# device selection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

train_dataset = datasets.MNIST(
    root = "../../data",
    train = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ]),
    download = True
)

test_dataset = datasets.MNIST(
    root = "../../data",
    train = False,
    transform = transforms.Compose([
        transforms.ToTensor()
    ]),
    download = True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False
)

model = CAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
total_step = len(train_loader)


for epoch in range(epochs):
    # training
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        outputs = model(images)
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

# Show originals and reconstruced images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Originals
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.axis("off")
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon[i].squeeze(), cmap='gray')
    plt.axis("off")
plt.show()
