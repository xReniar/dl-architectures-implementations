import torch
import torchvision
import torchvision.transforms as transforms

DATASET_CLASSES = {
    "CIFAR10": torchvision.datasets.CIFAR10,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "MNIST": torchvision.datasets.MNIST
}

def download_dataset(dataset_name: str, img_size: tuple):
    dataset_loader = DATASET_CLASSES[dataset_name]
    train_dataset = dataset_loader(
        root = '../data',
        train = True,
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ]),
        download = True
    )
    
    test_dataset = dataset_loader(
        root = '../data',
        train = False,
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
        ]),
        download = True
    )

    return train_dataset, test_dataset



def train_dataloader(train_dataset, batch_size:int):
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)
    return train_loader

def test_dataloader(test_dataset, batch_size:int):
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = batch_size,
                                              shuffle = True)
    return test_loader

def get_dataloader(dataset_name:str, batch_size:int, image_size:tuple):
    train_dataset, test_dataset = download_dataset(dataset_name, image_size)

    train_loader = train_dataloader(train_dataset, batch_size)
    test_loader = test_dataloader(test_dataset, batch_size)

    return train_loader, test_loader