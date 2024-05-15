import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers.neptune import NeptuneLogger
from neptune import ANONYMOUS_API_TOKEN
from neptune.types import File
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# define hyper-parameters
params = {
    "batch_size": 32,
    "linear": 32,
    "lr": 0.0005,
    "decay_factor": 0.9,
    "max_epochs": 3,
}


# (neptune) define model with logging (self.log)
class LitModel(pl.LightningModule):
    def __init__(self, linear, learning_rate, decay_factor):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.linear = linear
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.train_img_max = 10
        self.train_img = 0
        self.layer_1 = torch.nn.Linear(28 * 28, linear)
        self.layer_2 = torch.nn.Linear(linear, 20)
        self.layer_3 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor**epoch)
        return [optimizer], [scheduler]

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        self.validation_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_validation_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.validation_step_outputs:
            loss = np.append(loss, results_dict["loss"].detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("val/loss", loss.mean())
        self.log("val/acc", acc)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        for j in np.where(np.not_equal(y_true, y_pred))[0]:
            img = np.squeeze(x[j].cpu().detach().numpy())
            img[img < 0] = 0
            img = img / np.amax(img)
            neptune_logger.experiment["test/misclassified_images"].append(
                File.as_image(img),
                description=f"y_pred={y_pred[j]}, y_true={y_true[j]}",
            )

        self.test_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_test_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.test_step_outputs:
            loss = np.append(loss, results_dict["loss"].detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("test/loss", loss.mean())
        self.log("test/acc", acc)
        self.validation_step_outputs.clear()  # free memory


# define DataModule
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, normalization_vector):
        super().__init__()
        self.batch_size = batch_size
        self.normalization_vector = normalization_vector
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage):
        # transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vector[0], self.normalization_vector[1]),
            ]
        )
        if stage == "fit":
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=0)


# (neptune) log confusion matrix for classification
def log_confusion_matrix(lit_model, data_module):
    lit_model.freeze()
    test_data = data_module.test_dataloader()
    y_true = np.array([])
    y_pred = np.array([])
    for x, y in test_data:
        y = y.cpu().detach().numpy()
        y_hat = lit_model.forward(x).argmax(axis=1).cpu().detach().numpy()
        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax)
    neptune_logger.experiment["confusion_matrix"].upload(fig)


# create learning rate logger
lr_logger = LearningRateMonitor(logging_interval="epoch")

# create model checkpointing object
model_checkpoint = ModelCheckpoint(
    dirpath="my_model/checkpoints/",
    filename="{epoch:02d}",
    save_weights_only=True,
    save_top_k=2,
    save_last=True,
    monitor="val/loss",
    every_n_epochs=1,
)

# (neptune) create NeptuneLogger
neptune_logger = NeptuneLogger(
    api_key=ANONYMOUS_API_TOKEN,
    project="common/pytorch-lightning-integration",
    tags=["complex", "script"],
    log_model_checkpoints=True,
)

# (neptune) initialize a trainer and pass neptune_logger
trainer = pl.Trainer(
    logger=neptune_logger,
    callbacks=[lr_logger, model_checkpoint],
    log_every_n_steps=50,
    max_epochs=params["max_epochs"],
    enable_progress_bar=False,
)

# init model
model = LitModel(
    linear=params["linear"],
    learning_rate=params["lr"],
    decay_factor=params["decay_factor"],
)

# init datamodule
dm = MNISTDataModule(
    normalization_vector=((0.1307,), (0.3081,)),
    batch_size=params["batch_size"],
)

# (neptune) log model summary
neptune_logger.log_model_summary(model=model, max_depth=-1)

# (neptune) log hyper-parameters
neptune_logger.log_hyperparams(params=params)

# train and test the model, log metadata to the Neptune run
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

# (neptune) log confusion matrix
log_confusion_matrix(model, dm)
