from pathlib import Path

import helper_functions
import torch
from matplotlib import pyplot as plt
from model_functions import eval_model, train_model
from torch import nn, optim

import data

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model_v0.pth"

EPOCHS = 2


class Modelv0(nn.Module):
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


def main() -> None:
    model = Modelv0(
        input_shape=3,
        hidden_units=10,
        output_shape=len(data.dataset_classes),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.01, nesterov=True)

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

    helper_functions.plot_model_results(model_result)
    helper_functions.plot_classification_result(
        dataset=data.test_dataset_default,
        pred_labels=model_result["y_preds"],
        classes=data.dataset_classes,
        n=10,
        display_shape=True,
    )
    helper_functions.plot_confmat(
        dataset=data.test_dataset_default,
        model_result=model_result,
    )

    torch.save(obj=model.state_dict(), f=MODEL_PATH / MODEL_NAME)

    plt.show()


if __name__ == "__main__":
    main()
