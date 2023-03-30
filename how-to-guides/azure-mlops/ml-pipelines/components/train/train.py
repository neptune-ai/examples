import os
from pathlib import Path
from typing import List

import neptune
import pandas as pd
import xgboost as xgb
from mldesigner import Input, command_component
from neptune.integrations.xgboost import NeptuneCallback


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
    train_data_path: Input(type="uri_folder", description="Train data"),
):
    DATASET_PATH = os.path.join(train_data_path, "train_data.csv")
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
    model = model = xgb.XGBRegressor(random_state=42, callbacks=[neptune_callback])
    model.fit(X_train, y_train)
