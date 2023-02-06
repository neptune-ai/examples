import matplotlib.pyplot as plt
import neptune.new as neptune
import pytorch_lightning as pl
import seaborn as sns
from data_module import *
from model import *
from neptune.new.exceptions import NeptuneException
from neptune.new.types import File
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from utils import *

sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"


def main():
    params = {
        "seq_len": 8,
        "batch_size": 128,
        "criterion": nn.MSELoss(),
        "max_epochs": 10,
        "n_features": 1,
        "hidden_dim": 512,
        "n_layers": 5,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "year": 2010,
    }

    # (neptune) Create NeptuneLogger instance
    neptune_logger = NeptuneLogger(log_model_checkpoints=False)

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        callbacks=[early_stop, lr_logger],
        logger=neptune_logger,  # neptune integration
        accelerator="gpu",
    )

    dm = WalmartSalesDataModule(
        seq_len=params["seq_len"], num_workers=4, path="../../data", year=params["year"]
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

    # Train model
    trainer.fit(model, dm)

    # Manually save model checkpoint
    ckpt_name = "pre-trained"
    trainer.save_checkpoint(f"{ckpt_name}.ckpt")

    # (neptune) Log model checkpoint
    neptune_logger.experiment[f"training/model/checkpoints/{ckpt_name}"].upload(f"{ckpt_name}.ckpt")

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

    # (neptune) Log scores
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

    except NeptuneException:
        print(f"A model with the provided key {model_key} already exists in this project.")
        print("Creating a new model version...")
        model_version = neptune.init_model_version(
            model=f"{project_key}-{model_key}",
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
    model_ckpt_name = get_model_ckpt_name(neptune_logger.experiment)
    neptune_logger.experiment[f"training/model/checkpoints/{model_ckpt_name}"].download()

    # (neptune) Upload model checkpoint to Model registry
    model_version["checkpoint"].upload(f"{model_ckpt_name}.ckpt")


if __name__ == "__main__":
    main()
