import sys

import matplotlib.pyplot as plt
import neptune
import seaborn as sns
from data_module import *
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import NeptuneLogger
from model import *
from neptune.exceptions import ModelNotFound, NeptuneModelKeyAlreadyExistsError
from neptune.types import File

sys.path.append("../")

from utils import *

sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"

DATA_PATH = "../dataset"

params = {
    "seq_len": 8,
    "batch_size": 128,
    "criterion": nn.MSELoss(),
    "max_epochs": 1,
    "n_features": 1,
    "hidden_dim": 512,
    "n_layers": 5,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "year": 2011,
}

# (neptune) Get latest model version ID
with neptune.init_project(mode="read-only") as project:
    project_key = project["sys/id"].fetch()

model_key = "DL"
try:
    model = neptune.init_model(
        with_id=f"{project_key}-{model_key}",  # Your model ID here
    )
    model_versions_table = model.fetch_model_versions_table().to_pandas()
    latest_model_version_id = model_versions_table["sys/id"].tolist()[0]

except ModelNotFound:
    sys.exit(
        f"The model with the provided key `{model_key}` doesn't exist in the `{project_key}` project."
    )

# (neptune) Download the lastest model checkpoint from model registry
model_version = neptune.init_model_version(with_id=latest_model_version_id)

model_version["checkpoint"].download()

# (neptune) Create NeptuneLogger instance
neptune_logger = NeptuneLogger(
    tags=["LSTM", "fine-tuned", "walmart-sales"],
    name="LSTM finetuning",
    log_model_checkpoints=False,
)

early_stop = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()

trainer = Trainer(
    max_epochs=params["max_epochs"],
    callbacks=[early_stop, lr_logger],
    logger=neptune_logger,  # neptune integration
    accelerator="auto",
    enable_progress_bar=False,
)

dm = WalmartSalesDataModule(
    seq_len=params["seq_len"], num_workers=0, path=DATA_PATH, year=params["year"]
)

model = LSTMRegressor(
    n_features=params["n_features"],
    hidden_dim=params["hidden_dim"],
    criterion=params["criterion"],
    n_layers=params["n_layers"],
    dropout=params["dropout"],
    learning_rate=params["learning_rate"],
    seq_len=params["seq_len"],
)

model = LSTMRegressor.load_from_checkpoint("checkpoint.ckpt")

# Train model
trainer.fit(model, dm)

# Manually save model checkpoint
ckpt_name = "fine-tuned"
trainer.save_checkpoint(f"{ckpt_name}.ckpt")

# (neptune) Log model checkpoint
neptune_logger.experiment["training/model/checkpoints"][ckpt_name].upload(f"{ckpt_name}.ckpt")

# Test model
test_loader = dm.test_dataloader()
predictions, values = model.predict(test_loader)
df_result = format_predictions(predictions, values, dm.scaler)

preds_plot = sns.lineplot(data=df_result)

# (neptune) Log predictions visualizations
neptune_logger.experiment["training/plots/ypred_vs_y_valid"].upload(
    File.as_image(preds_plot.figure)
)

val_metrics = calculate_metrics(df_result)

# (neptune) Log validation scores
neptune_logger.experiment["training/val"] = val_metrics

# (neptune) Initializing a Model and Model version
model_key = "DL"
project_key = neptune_logger.experiment["sys/id"].fetch().split("-")[0]

try:
    model = neptune.init_model(
        key=model_key,
    )

    print("Creating a new model version...")
    model_version = neptune.init_model_version(model=f"{project_key}-{model_key}")

except NeptuneModelKeyAlreadyExistsError:
    print(f"A model with the provided key `{model_key}` already exists in this project.")
    print("Creating a new model version...")
    model_version = neptune.init_model_version(
        model=f"{project_key}-{model_key}", name="LSTM-finetuned"
    )

# (neptune) Log model version details to run
neptune_logger.experiment["model_version/id"] = model_version["sys/id"].fetch()
neptune_logger.experiment["model_version/model_id"] = model_version["sys/model_id"].fetch()
neptune_logger.experiment["model_version/url"] = model_version.get_url()

# (neptune) Log run details
model_version["run/id"] = neptune_logger.experiment["sys/id"].fetch()
model_version["run/name"] = neptune_logger.experiment["sys/name"].fetch()
model_version["run/url"] = neptune_logger.experiment.get_url()

# (neptune) Log validation scores from run
model_version["training/val"] = neptune_logger.experiment["training/val"].fetch()

# (neptune) Download model checkpoint from Run
neptune_logger.experiment.wait()
model_ckpt_name = list(
    neptune_logger.experiment.get_structure()["training"]["model"]["checkpoints"].keys()
)[-1]
neptune_logger.experiment[f"training/model/checkpoints/{model_ckpt_name}"].download()

# (neptune) Upload model checkpoint to Model registry
model_version["checkpoint"].upload(f"{model_ckpt_name}.ckpt")
