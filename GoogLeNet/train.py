import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import GoogLeNet


# device selection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"

# hyperparameters
img_size = (224, 224)
batch_size = 30
lr = 0.001
alpha = 0.3
num_epochs = 20

train_dataset = datasets.CIFAR10(
    root = "../data",
    train = True,
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
    download = True
)

test_dataset = datasets.CIFAR10(
    root = "../data",
    train = False,
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
    download = True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

model = GoogLeNet(10, "v1").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

total_step = len(train_loader)
for epoch in range(num_epochs):
    # training
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        aux1, aux2, output = model(images)
        loss_main = criterion(output, labels)
        loss_aux1 = criterion(aux1, labels)
        loss_aux2 = criterion(aux2, labels)

        loss = loss_main + (alpha * loss_aux1) + (alpha * loss_aux2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # testing
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            _, _, output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))