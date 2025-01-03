import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loader import get_dataloader
from models.AlexNet.model import AlexNet
from models.Resnet.model import Resnet
from models.ResNeXt50.model import ResNeXt50
from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model:nn.Module, loss_fn, optimizer:torch.optim.Optimizer, batch_size:int, num_epochs:int):
    # get dataloaders
    train_loader, test_loader = get_dataloader("CIFAR10",batch_size, (224,224))

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        # training
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

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # testing
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))

if __name__ == "__main__":
    # define variables
    batch_size = 16
    num_classes = 10
    lr = 0.001
    num_epochs = 20

    # define hyperparameters
    #model = Resnet(34, num_classes).to(device)
    #model = AlexNet(num_classes).to(device) 
    model = ResNeXt50(32, 4, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.005, momentum=0.9)
    
    train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=batch_size,
        num_epochs=num_epochs
    )