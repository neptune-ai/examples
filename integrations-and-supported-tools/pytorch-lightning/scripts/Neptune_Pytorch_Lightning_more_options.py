import os

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import torch
import torch.nn.functional as F
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger

# define hyper-parameters
PARAMS = {
    "batch_size": 64,
    "linear": 32,
    "lr": 0.005,
    "decay_factor": 0.995,
    "max_epochs": 15,
}


# (neptune) define model with logging (self.log)
class LitModel(pl.LightningModule):
    def __init__(self, linear, learning_rate, decay_factor):
        super().__init__()
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
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor ** epoch)
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

        return {"loss": loss,
                "y_true": y_true,
                "y_pred": y_pred}

    def training_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("train/epoch/loss", loss.mean())
        self.log("train/epoch/acc", acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        return {"loss": loss,
                "y_true": y_true,
                "y_pred": y_pred}

    def validation_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("val/loss", loss.mean())
        self.log("val/acc", acc)

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
            neptune_logger.experiment["test/misclassified_images"].log(
                neptune.types.File.as_image(img),
                description="y_pred={}, y_true={}".format(y_pred[j], y_true[j]),
            )

        return {"loss": loss,
                "y_true": y_true,
                "y_pred": y_pred}

    def test_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("test/loss", loss.mean())
        self.log("test/acc", acc)


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
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.normalization_vector[0],
                                                             self.normalization_vector[1])])
        if stage == "fit":
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=1)
        return mnist_test


# (neptune) log confusion matrix for classification
def log_confusion_matrix(lit_model, data_module):
    lit_model.freeze()
    test_data = data_module.test_dataloader()
    y_true = np.array([])
    y_pred = np.array([])
    for i, (x, y) in enumerate(test_data):
        y = y.cpu().detach().numpy()
        y_hat = lit_model.forward(x).argmax(axis=1).cpu().detach().numpy()
        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax)
    neptune_logger.experiment["confusion_matrix"].upload(neptune.types.File.as_image(fig))


# create learning rate logger
lr_logger = LearningRateMonitor(logging_interval="epoch")

# create model checkpointing object
model_checkpoint = ModelCheckpoint(
    dirpath="my_model/checkpoints/",
    filename="{epoch:02d}",
    save_weights_only=True,
    save_top_k=3,
    save_last=True,
    monitor="val/loss",
    every_n_epochs=1
)

# (neptune) create NeptuneLogger
neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project="common/pytorch-lightning-integration",
    tags=["complex", "showcase"],
)

# (neptune) initialize a trainer and pass neptune_logger
trainer = pl.Trainer(
    logger=neptune_logger,
    callbacks=[lr_logger, model_checkpoint],
    log_every_n_steps=50,
    max_epochs=PARAMS["max_epochs"],
    track_grad_norm=2,
)

# init model
model = LitModel(
    linear=PARAMS["linear"],
    learning_rate=PARAMS["lr"],
    decay_factor=PARAMS["decay_factor"],
)

# init datamodule
dm = MNISTDataModule(
    normalization_vector=((0.1307,), (0.3081,)),
    batch_size=PARAMS["batch_size"],
)

# (neptune) log model summary
neptune_logger.log_model_summary(model=model, max_depth=-1)

# (neptune) log hyper-parameters
neptune_logger.log_hyperparams(params=PARAMS)

# train and test the model, log metadata to the Neptune run
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

# (neptune) log confusion matrix
log_confusion_matrix(model, dm)
