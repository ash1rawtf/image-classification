from pathlib import Path

import helper_functions
import torch
from config import logger
from matplotlib import pyplot as plt
from model_v0 import Modelv0, eval_model, train_model
from torch import nn, optim
from torch.utils.data import DataLoader

import data

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
EPOCHS = 2


def run_model_v0(
    model_name: str,
    model_desc: str,
    dataloaders: dict[str: DataLoader],
) -> dict[str, list[float] | float | str]:
    train_dataloader = dataloaders["train_dataloader"]
    test_dataloader = dataloaders["test_dataloader"]

    logger.info(f"Running {model_desc}...")

    model = Modelv0(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_dataloader),
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
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
    ))

    model_result["model_name"] = model_name
    model_result["model_desc"] = model_desc

    torch.save(obj=model.state_dict(), f=MODEL_PATH / f"{model_name}.pth")
    logger.info(f"{model_desc} state dict has been successfully saved.")

    return model_result


def main() -> None:
    model_v0_transform_default_result = run_model_v0(
        model_name="model_v0",
        model_desc="SGD(lr=0.01, momentum=0.01, nesterov=True) | Default dataset",
        dataloaders={
            "train_dataloader": data.train_dataloader_default,
            "test_dataloader": data.test_dataloader_default,
        },
    )

    model_v0_transform_v0_result = run_model_v0(
        model_name="model_v0_transform_v0",
        model_desc="SGD(lr=0.01, momentum=0.01, nesterov=True) | RandomHorizontalFlip(p=0.5)",
        dataloaders={
            "train_dataloader": data.train_dataloader_transform_v0,
            "test_dataloader": data.test_dataloader_transform_v0,
        },
    )

    helper_functions.plot_models_result([
        model_v0_transform_default_result,
        model_v0_transform_v0_result,
    ])

    plt.show()


if __name__ == "__main__":
    main()
