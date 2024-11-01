from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 32

train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        transform=ToTensor(),
        target_transform=None,
        download=True,
    )

test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        transform=ToTensor(),
        target_transform=None,
        download=True,
    )

train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

dataset_classes = train_dataset.classes
