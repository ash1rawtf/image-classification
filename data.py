import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=None,
    download=True,
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
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

train_transform_v0 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

transform_v0_train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    transform=train_transform_v0,
    target_transform=None,
    download=True,
)

transform_v0_test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=None,
    download=True,
)

transform_v0_train_dataloader = DataLoader(
    dataset=transform_v0_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

transform_v0_test_dataloader = DataLoader(
    dataset=transform_v0_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
