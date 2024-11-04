from timeit import default_timer

import helper_functions
import torch
from config import logger
from torch import nn, optim
from torch.utils.data import DataLoader

# from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        train_acc += helper_functions.accuracy_fn(y_preds=y_preds.argmax(dim=1), y_true=y)
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
            test_acc += helper_functions.accuracy_fn(y_preds=y_preds.argmax(dim=1), y_true=y)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
) -> dict[str, list[float]]:
    model_result = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    train_time_start = default_timer()

    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
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

        logger.info(f"Epoch: {epoch + 1}/{epochs} | Loss: {train_loss:.5f} | "
                    f"Acc: {train_acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")

        model_result["train_loss"].append(train_loss.item())
        model_result["train_acc"].append(train_acc)
        model_result["test_loss"].append(test_loss.item())
        model_result["test_acc"].append(test_acc)

    train_time_end = default_timer()

    logger.info(f"Model train time: {train_time_end - train_time_start:.3f} seconds")

    model_result["train_time"] = train_time_end - train_time_start

    return model_result


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
            eval_acc += helper_functions.accuracy_fn(y_preds=y_preds.argmax(dim=1), y_true=y)
            y_preds_list.append(torch.softmax(y_preds.squeeze(), dim=0).argmax(dim=1).cpu())

        eval_loss /= len(dataloader)
        eval_acc /= len(dataloader)

    logger.info(f"Model eval loss: {eval_loss:.5f} | Acc: {eval_acc:.2f}%")

    return {
        "eval_loss": eval_loss.item(),
        "eval_acc": eval_acc,
        "y_preds": torch.cat(y_preds_list),
    }
