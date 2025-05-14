import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"