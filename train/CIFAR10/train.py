import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loader import get_dataloader
from models.AlexNet.model import AlexNet
from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model:nn.Module, loss_fn, optimizer:torch.optim.Optimizer, batch_size:int, num_epochs:int):
    # get dataloaders
    train_loader, test_loader = get_dataloader("CIFAR10",batch_size, (227,227))

    # training
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 400 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # testing

if __name__ == "__main__":
    # define variables
    batch_size = 64
    num_classes = 10
    lr = 0.001
    num_epochs = 10

    # define hyperparameters
    model = AlexNet(num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=batch_size,
        num_epochs=num_epochs
    )