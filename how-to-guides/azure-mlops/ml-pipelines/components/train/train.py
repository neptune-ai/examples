import os
from pathlib import Path
from typing import List

import neptune
import pandas as pd
import xgboost as xgb
from mldesigner import Input, command_component
from neptune.integrations.xgboost import NeptuneCallback
from utils import get_train_data


def load_xgboost_model(callbacks: List = None, random_state: int = 42):
    model = xgb.XGBRegressor(random_state=random_state, callbacks=callbacks)
    return model


@command_component(
    name="train",
    display_name="Train model",
    description="Train model",
    version="0.1",
    environment=dict(
        conda=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def train_component(
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
