from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "data"
DOWNLOAD = not (Path(ROOT_DIR) / datasets.CIFAR10.base_folder).exists()
BATCH_SIZE = 32

train_dataset_default = datasets.CIFAR10(
    root=ROOT_DIR,
    train=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=None,
    download=DOWNLOAD,
)

test_dataset_default = datasets.CIFAR10(
    root=ROOT_DIR,
    train=False,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=None,
    download=DOWNLOAD,
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

train_dataset_transform_v0 = datasets.CIFAR10(
    root=ROOT_DIR,
    train=True,
    transform=v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=None,
    download=DOWNLOAD,
)

test_dataset_transform_v0 = datasets.CIFAR10(
    root=ROOT_DIR,
    train=False,
    transform=v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=None,
    download=DOWNLOAD,
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
