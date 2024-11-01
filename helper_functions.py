import random

import torch
import torch.utils
import torch.utils.data
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix


def accuracy_fn(y_preds: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.eq(y_preds, y_true).sum().item() / len(y_preds) * 100


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


def plot_confmat(
    dataset: torch.utils.data.Dataset,
    model_result: dict[str, list[float] | float | str],
) -> None:
    confmat = ConfusionMatrix(num_classes=len(dataset.classes), task="multiclass")
    confmat_tensor = confmat(preds=model_result["y_preds"], target=torch.Tensor(dataset.targets))

    figure, _ = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=dataset.classes,
        figsize=(10, 7),
    )

    figure.suptitle(f"{model_result["model_name"]} confusion matrix")
