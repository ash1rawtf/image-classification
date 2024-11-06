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


def plot_model_results(model_result: dict[str, list[float] | float | str]) -> None:
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


def plot_models_result(models_result: list[dict[str, list[float] | float | str]]) -> None:
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), num="Image classification model results")

    models = []
    models_train_time = []
    models_avg_train_time_per_epoch = []

    for model_result in models_result:
        train_loss = model_result["train_loss"]
        train_acc = model_result["train_acc"]
        test_loss = model_result["test_loss"]
        test_acc = model_result["test_acc"]
        epochs = len(train_loss)

        axs[0, 0].plot(range(1, epochs + 1), train_loss, label=model_result["model_name"])
        axs[0, 1].plot(range(1, epochs + 1), test_loss, label=model_result["model_name"])
        axs[0, 2].plot(range(1, epochs + 1), train_acc, label=model_result["model_name"])
        axs[0, 3].plot(range(1, epochs + 1), test_acc, label=model_result["model_name"])

        models.append(model_result["model_name"])
        models_train_time.append(model_result["train_time"])
        models_avg_train_time_per_epoch.append(model_result["avg_train_time_per_epoch"])

    axs[0, 0].set_title("Train loss")
    axs[0, 1].set_title("Test loss")
    axs[0, 2].set_title("Train accuracy")
    axs[0, 3].set_title("Test accuracy")

    axs[0, 0].set(xlabel="Epochs")
    axs[0, 1].set(xlabel="Epochs")
    axs[0, 2].set(xlabel="Epochs")
    axs[0, 3].set(xlabel="Epochs")

    bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    axs[1, 0].set_title("Models train time")
    axs[1, 0].bar_label(
        axs[1, 0].barh(models, models_train_time, color=bar_colors),
        fmt="%.2f",
        label_type="center",
    )
    axs[1, 0].set_xlim(right=max(models_train_time) * 1.1)
    axs[1, 0].set_xlabel("Time, s.")
    axs[1, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axs[1, 1].set_title("Models avarage train time per epoch")
    axs[1, 1].bar_label(
        axs[1, 1].barh(models, models_avg_train_time_per_epoch, color=bar_colors),
        fmt="%.2f",
        label_type="center",
    )
    axs[1, 1].set_xlim(right=max(models_avg_train_time_per_epoch) * 1.1)
    axs[1, 1].set_xlabel("Time, s.")
    axs[1, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.delaxes(axs[1, 2])
    fig.delaxes(axs[1, 3])

    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center")
