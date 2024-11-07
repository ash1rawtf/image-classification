from pathlib import Path

import helper_functions
import torch
from config import logger
from matplotlib import pyplot as plt
from model_v0 import Modelv0, eval_model, train_model
from torch import nn, optim

import data

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
EPOCHS = 2


def run_model_v0() -> dict[str, list[float] | float | str]:
    model_name = "model_v0.pth"
    logger.info("Running Modelv0...")

    model = Modelv0(
        input_shape=3,
        hidden_units=10,
        output_shape=len(data.dataset_classes),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=0.01,
        momentum=0.01,
        nesterov=True,
    )

    model_result = train_model(
        model=model,
        train_dataloader=data.train_dataloader_default,
        test_dataloader=data.test_dataloader_default,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=data.test_dataloader_default,
        loss_fn=loss_fn,
    ))

    model_result["model_name"] = "SGD(lr=0.01, momentum=0.01, nesterov=True) | Default dataset"

    torch.save(obj=model.state_dict(), f=MODEL_PATH / model_name)
    logger.info("Model_v0 state dict has been successfully saved.")

    return model_result


def run_model_v0_w_transform_v0() -> dict[str, list[float] | float | str]:
    model_name = "model_v0_w_transform_v0.pth"
    logger.info("Running Modelv0 with transform_v0...")

    model = Modelv0(
        input_shape=3,
        hidden_units=10,
        output_shape=len(data.dataset_classes),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=0.01,
        momentum=0.01,
        nesterov=True,
    )

    model_result = train_model(
        model=model,
        train_dataloader=data.train_dataloader_transform_v0,
        test_dataloader=data.test_dataloader_transform_v0,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=data.test_dataloader_transform_v0,
        loss_fn=loss_fn,
    ))

    model_result["model_name"] = "SGD(lr=0.01, momentum=0.01, nesterov=True) | RandomHorizontalFlip(p=0.5)"

    torch.save(obj=model.state_dict(), f=MODEL_PATH / model_name)
    logger.info("Modelv0 with transform_v0 state dict has been successfully saved.")

    return model_result


def main() -> None:
    model_v0_result = run_model_v0()
    model_v0_w_transform_v0_result = run_model_v0_w_transform_v0()

    helper_functions.plot_models_result([
        model_v0_result,
        model_v0_w_transform_v0_result,
    ])

    plt.show()


if __name__ == "__main__":
    main()
