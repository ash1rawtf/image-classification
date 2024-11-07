import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

train_dataset_default = datasets.CIFAR10(
    root="data",
    train=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=None,
    # download=True,
)

test_dataset_default = datasets.CIFAR10(
    root="data",
    train=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=None,
    # download=True,
)

train_dataloader_default = DataLoader(
    dataset=train_dataset_default,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dataloader_default = DataLoader(
    dataset=test_dataset_default,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

dataset_classes = train_dataset_default.classes

train_transform_v0 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_dataset_transform_v0 = datasets.CIFAR10(
    root="data",
    train=True,
    transform=train_transform_v0,
    target_transform=None,
    # download=True,
)

test_dataset_transform_v0 = datasets.CIFAR10(
    root="data",
    train=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=None,
    # download=True,
)

train_dataloader_transform_v0 = DataLoader(
    dataset=train_dataset_transform_v0,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dataloader_transform_v0 = DataLoader(
    dataset=test_dataset_transform_v0,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
