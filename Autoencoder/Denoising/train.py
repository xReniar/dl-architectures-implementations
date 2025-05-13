import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import DAE


# device selection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

train_dataset = datasets.ImageNet(
    root = "../../data",
    train = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ]),
    download = True
)

test_dataset = datasets.ImageNet(
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

model = DAE()