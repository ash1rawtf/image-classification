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


def run_model(
    model: nn.Module,
    model_name: str,
    model_desc: str,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    dataloaders: dict[str: DataLoader],
) -> dict[str, list[float] | float | str]:
    logger.info(f'Running "{model_desc}"...')

    model_result = train_model(
        model=model,
        train_dataloader=dataloaders["train_dataloader"],
        test_dataloader=dataloaders["test_dataloader"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=dataloaders["test_dataloader"],
        loss_fn=loss_fn,
    ))

    model_result["model_name"] = model_name
    model_result["model_desc"] = model_desc

    torch.save(obj=model.state_dict(), f=MODEL_PATH / f"{model_name}.pth")
    logger.info(f'"{model_desc}" state dict has been successfully saved.')

    return model_result


def main() -> None:
    model_v0_ex1 = Modelv0(
        input_shape=3,
        hidden_units=10,
        output_shape=len(data.train_dataset_default),
    ).to(device)

    model_v0_transform_default_ex1_result = run_model(
        model=model_v0_ex1,
        model_name="model_v0_transform_default_ex1",
        model_desc="SGD(lr=0.01) | Default dataset",
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(
            params=model_v0_ex1.parameters(),
            lr=0.01,
        ),
        dataloaders={
            "train_dataloader": data.train_dataloader_default,
            "test_dataloader": data.test_dataloader_default,
        },
    )

    del model_v0_ex1
    torch.cuda.empty_cache()

    helper_functions.plot_models_result([
        model_v0_transform_default_ex1_result,
    ])

    plt.show()


if __name__ == "__main__":
    main()
