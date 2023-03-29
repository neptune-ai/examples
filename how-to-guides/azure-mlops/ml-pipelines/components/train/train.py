import os
from pathlib import Path

import neptune
import pandas as pd
from mldesigner import Input, Output, command_component
from model import load_xgboost_model
from neptune.integrations.xgboost import NeptuneCallback
from utils import get_train_data


@command_component(
    name="train",
    description="Train model",
    version="0.1",
    environment=dict(
        conda=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def train(
    train_data: Input(type="uri_folder", description="Train data"),
):
    DATASET_PATH = os.path.join(train_data, "train_data.csv")
    train_df = pd.read_csv(DATASET_PATH)

    # Get train data
    X_train, y_train = train_df.drop(["Weekly_Sales"], axis=1), train_df.Weekly_Sales

    # (neptune) Initialize Neptune run
    run = neptune.init_run(
        tags=["MLOps", "baseline", "xgboost", "walmart-sales"],
        name="XGBoost",
    )

    neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])

    # Train model
    model = load_xgboost_model(callbacks=[neptune_callback], random_state=42)
    model.fit(X_train, y_train)
