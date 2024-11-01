from pathlib import Path

import data
import helper_functions
import torch
from config import logger
from image_classification_v0 import ImageClassificationModelv0, eval_model, train_model
from matplotlib import pyplot as plt
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
EPOCHS = 2


def run_image_classification_v0() -> None:
    model_name = "image_classification_v0.pth"
    logger.info("Running ImageClassificationModelv0 model...")

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

    helper_functions.plot_model_results(model_result)
    helper_functions.plot_classification_result(
        dataset=data.test_dataset,
        pred_labels=model_result["y_preds"],
        classes=data.train_dataset.classes,
        n=10,
        display_shape=True,
    )
    helper_functions.plot_confmat(
        dataset=data.test_dataset,
        model_result=model_result,
    )

    torch.save(obj=model.state_dict(), f=MODEL_PATH / model_name)
    logger.info(f"Model {model_name} state dict was successfully saved!")

    plt.show()


def main() -> None:
    run_image_classification_v0()


if __name__ == "__main__":
    main()
