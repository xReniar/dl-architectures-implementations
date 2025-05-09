import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import ResNeXt50


# device selection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

# defining image size
img_size = (224, 224)

# define iperparameters
batch_size = 30
num_classes = 10
lr = 0.001
num_epochs = 20

train_dataset = torchvision.datasets.CIFAR10(
    root = "../../data",
    train = True,
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
    download = True
)

test_dataset = torchvision.datasets.CIFAR10(
    root = "../../data",
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

model = ResNeXt50(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params = model.parameters(),
    lr = lr,
    weight_decay= 0.005,
    momentum=0.9
)

# training process
total_step = len(train_loader)
for epoch in range(num_epochs):
    # training
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

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

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))