import random
from pathlib import Path
from timeit import default_timer

import torch
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "image_classification_v0.pth"

BATCH_SIZE = 32
EPOCHS = 2


class ImageClassificationModelv0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 8 * 8, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def accuracy_fn(y_preds: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.eq(y_preds, y_true).sum().item() / len(y_preds) * 100


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
) -> tuple[float, float]:
    train_loss, train_acc = 0, 0

    model.train()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        loss = loss_fn(y_preds, y)
        train_loss += loss
        train_acc += accuracy_fn(y_preds=y_preds.argmax(dim=1), y_true=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
) -> tuple[float, float]:
    test_loss, test_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_preds = model(X)
            loss = loss_fn(y_preds, y)
            test_loss += loss
            test_acc += accuracy_fn(y_preds=y_preds.argmax(dim=1), y_true=y)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def eval_model(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
) -> dict[str, str | float | torch.Tensor]:
    eval_loss, eval_acc = 0, 0
    y_preds_list = []

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_preds = model(X)
            eval_loss += loss_fn(y_preds, y)
            eval_acc += accuracy_fn(y_preds=y_preds.argmax(dim=1), y_true=y)
            y_preds_list.append(torch.softmax(y_preds.squeeze(), dim=0).argmax(dim=1).cpu())

        eval_loss /= len(dataloader)
        eval_acc /= len(dataloader)

    print(f"\nModel eval loss: {eval_loss:.5f} | Acc: {eval_acc:.2f}%")

    return {
        "model_name": model.__class__.__name__,
        "eval_loss": eval_loss.item(),
        "eval_acc": eval_acc,
        "y_preds": torch.cat(y_preds_list),
    }


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
) -> dict[str, list[float]]:
    model_result = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    train_time_start = default_timer()

    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
        )

        print(f"\nEpoch: {epoch}/{EPOCHS} | Loss: {train_loss:.5f} | Acc: {train_acc:.2f}% | "
              f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

        model_result["train_loss"].append(train_loss.item())
        model_result["train_acc"].append(train_acc)
        model_result["test_loss"].append(test_loss.item())
        model_result["test_acc"].append(test_acc)

    train_time_end = default_timer()

    print(f"Model train time: {train_time_end - train_time_start:.3f} seconds")

    return model_result


def plot_classification_result(
        dataset: torch.utils.data.Dataset,
        pred_labels: list[int],
        classes: list[str],
        n: int = 10,
        display_shape: bool = True,
) -> None:
    if n > 10:
        n = 10
        display_shape = False

    random_sample_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(15, 8), num=dataset.filename)
    plt.subplots_adjust(wspace=0.2, hspace=0)

    for i, current_sample in enumerate(random_sample_idx):
        targ_image, targ_label = dataset[current_sample][0], dataset[current_sample][1]
        pred_label = pred_labels[current_sample]

        targ_image_adjusted = targ_image.permute(1, 2, 0)

        plt.subplot(2, 5, i + 1)
        plt.imshow(targ_image_adjusted)
        plt.axis("off")
        title = f"Target class: {classes[targ_label]}"
        title += f"\nPred class: {classes[pred_label]}"
        if display_shape:
            title += f"\n{targ_image_adjusted.shape}"
        if targ_label == pred_label:
            plt.title(title, c="g")
        else:
            plt.title(title, c="r")


def plot_model_results(model_result: dict[str, list[float]]) -> None:
    train_loss = model_result["train_loss"]
    train_acc = model_result["train_acc"]
    test_loss = model_result["test_loss"]
    test_acc = model_result["test_acc"]
    epochs = len(train_loss)

    plt.figure(figsize=(10, 5), num="Image classification model results")
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss, label="Train loss")
    plt.plot(range(epochs), test_loss, label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_acc, label="Train accuracy")
    plt.plot(range(epochs), test_acc, label="Test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def main() -> None:
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

    model = ImageClassificationModelv0(
        input_shape=3,
        hidden_units=10,
        output_shape=len(dataset_classes),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.01, nesterov=True)

    model_result = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
    ))

    plot_model_results(model_result)
    plot_classification_result(
        dataset=test_dataset,
        pred_labels=model_result["y_preds"],
        classes=dataset_classes,
        n=10,
        display_shape=True,
    )

    confmat = ConfusionMatrix(num_classes=len(dataset_classes), task="multiclass")
    confmat_tensor = confmat(preds=model_result["y_preds"], target=torch.Tensor(test_dataset.targets))

    figure, _ = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=dataset_classes,
        figsize=(10, 7),
    )

    figure.suptitle(f"{model_result["model_name"]} confusion matrix")

    torch.save(obj=model.state_dict(), f=MODEL_PATH / MODEL_NAME)

    plt.show()


if __name__ == "__main__":
    main()
