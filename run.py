from pathlib import Path

import helper_functions
import torch
from config import logger
from image_classification_v0 import ImageClassificationModelv0, eval_model, train_model
from matplotlib import pyplot as plt
from torch import nn, optim

import data

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
EPOCHS = 2


def run_image_classification_model_v0() -> dict[str, list[float] | float | str]:
    model_name = "image_classification_v0.pth"
    logger.info("Running ImageClassificationModelv0...")

    model = ImageClassificationModelv0(
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
        train_dataloader=data.train_dataloader,
        test_dataloader=data.test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=data.test_dataloader,
        loss_fn=loss_fn,
    ))

    model_result["model_name"] = "SGD(lr=0.01, momentum=0.01, nesterov=True) | Default dataset"

    torch.save(obj=model.state_dict(), f=MODEL_PATH / model_name)
    logger.info("Models ImageClassificationModelv0 state dict was successfully saved!")

    return model_result


def run_image_classification_model_v0_w_transform_v0() -> dict[str, list[float] | float | str]:
    model_name = "image_classification_model_v0_w_transform_v0.pth"
    logger.info("Running ImageClassificationModelv0 with transform_v0...")

    model = ImageClassificationModelv0(
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
        train_dataloader=data.transform_v0_train_dataloader,
        test_dataloader=data.transform_v0_test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    model_result.update(eval_model(
        model=model,
        dataloader=data.transform_v0_test_dataloader,
        loss_fn=loss_fn,
    ))

    model_result["model_name"] = "SGD(lr=0.01, momentum=0.01, nesterov=True) | RandomHorizontalFlip(p=0.5)"

    torch.save(obj=model.state_dict(), f=MODEL_PATH / model_name)
    logger.info("Models ImageClassificationModelv0 with transform_v0 state dict was successfully saved!")

    return model_result


def main() -> None:
    image_classification_model_v0_result = run_image_classification_model_v0()
    image_classification_model_v0_w_transform_v0_result = run_image_classification_model_v0_w_transform_v0()

    helper_functions.plot_models_result([
        image_classification_model_v0_result,
        image_classification_model_v0_w_transform_v0_result,
    ])

    plt.show()


if __name__ == "__main__":
    main()
