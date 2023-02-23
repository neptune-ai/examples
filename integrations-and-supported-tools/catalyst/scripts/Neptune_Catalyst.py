import os
from collections import OrderedDict

import neptune
from catalyst import dl
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Prepare hparams
my_hparams = {"lr": 0.07, "batch_size": 32}

# Prepare model, criterion, optimizer and data loaders
model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), my_hparams["lr"])
loaders = OrderedDict(
    {
        "training": DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
            batch_size=my_hparams["batch_size"],
        ),
        "validation": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
            batch_size=my_hparams["batch_size"],
        ),
    }
)

# Create runner
my_runner = dl.SupervisedRunner()

# Create NeptuneLogger
neptune_logger = dl.NeptuneLogger(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/catalyst-integration",
    tags=["docs-example", "quickstart"],
)

# Train the model, pass neptune_logger
my_runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loggers={"neptune": neptune_logger},
    loaders=loaders,
    num_epochs=5,
    callbacks=[
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=[1]),
        dl.CheckpointCallback(
            logdir="checkpoints",
            loader_key="validation",
            metric_key="loss",
            minimize=True,
        ),
    ],
    hparams=my_hparams,
    valid_loader="validation",
    valid_metric="loss",
    minimize_valid_metric=True,
)

# Log best model
my_runner.log_artifact(
    path_to_artifact="./checkpoints/model.best.pth",
    tag="best_model",
    scope="experiment",
)
