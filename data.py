from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

train_transform_v0 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
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
    transform=ToTensor(),
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
