import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger

# define hyper-parameters
PARAMS = {
    "batch_size": 32,
    "lr": 0.007,
    "max_epochs": 15,
}


# (neptune) define LightningModule with logging (self.log)
class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("metrics/batch/loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/batch/acc", acc)

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def training_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss", loss.mean())
        self.log("metrics/epoch/acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=PARAMS["lr"])


# init model
mnist_model = MNISTModel()

# init DataLoader from MNIST dataset
train_ds = MNIST(
    os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_ds, batch_size=PARAMS["batch_size"])

# (neptune) create NeptuneLogger
neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project="common/pytorch-lightning-integration",
    tags=["simple", "showcase"],
    log_model_checkpoints=False,
)

# (neptune) initialize a trainer and pass neptune_logger
trainer = Trainer(
    logger=neptune_logger,
    max_epochs=PARAMS["max_epochs"],
)

# (neptune) log hyper-parameters
neptune_logger.log_hyperparams(params=PARAMS)

# train the model log metadata to the Neptune run
trainer.fit(mnist_model, train_loader)
