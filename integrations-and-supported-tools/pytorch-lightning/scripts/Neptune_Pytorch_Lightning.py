import os

import neptune
import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers.neptune import NeptuneLogger
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# define hyper-parameters
params = {
    "batch_size": 8,
    "lr": 0.005,
    "max_epochs": 2,
}


# (neptune) define LightningModule with logging (self.log)
class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/batch/loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("train/batch/acc", acc)

        self.training_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_train_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.training_step_outputs:
            loss = np.append(loss, results_dict["loss"].detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("train/epoch/loss", loss.mean())
        self.log("train/epoch/acc", acc)
        self.training_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=params["lr"])


# init model
mnist_model = MNISTModel()

# init DataLoader from MNIST dataset
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=params["batch_size"])

# (neptune) create NeptuneLogger
neptune_logger = NeptuneLogger(
    api_key=neptune.ANONYMOUS_API_TOKEN,
    project="common/pytorch-lightning-integration",
    tags=["simple", "script"],
    log_model_checkpoints=True,
)

# (neptune) initialize a trainer and pass neptune_logger
trainer = Trainer(
    logger=neptune_logger,
    max_epochs=params["max_epochs"],
    enable_progress_bar=False,
)

# (neptune) log hyper-parameters
neptune_logger.log_hyperparams(params=params)

# train the model log metadata to the Neptune run
trainer.fit(mnist_model, train_loader)
