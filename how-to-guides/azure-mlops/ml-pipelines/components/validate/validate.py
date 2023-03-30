import os
from pathlib import Path
from typing import List

import neptune
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from mldesigner import Input, command_component
from model import load_xgboost_model
from neptune.types import File
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import get_train_data


def load_xgboost_model(checkpoint, random_state: int = 42):
    model = xgb.XGBRegressor(random_state=random_state)
    model.load_model(checkpoint)
    return model


@command_component(
    name="validate",
    display_name="Validate model",
    description="Validate model",
    version="0.1",
    environment=dict(
        conda=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def validate_component(test_data: Input(type="uri_folder", description="Test data")):
    # (neptune) Initialize Neptune run
    run = neptune.init_run(
        tags=["MLOps", "baseline", "xgboost", "walmart-sales"],
        name="XGBoost",
    )
    # Load data
    DATASET_PATH = os.path.join(test_data, "validation_data.csv")
    df_valid = pd.read_csv(DATASET_PATH)
    # Get train data
    X_valid, y_valid = df_valid.drop(["Weekly_Sales"], axis=1), df_valid.Weekly_Sales

    # Load model checkpoint from model registry

    PATH = ""  # TODO: Get model from  model registry
    model = xgb.XGBRegressor(random_state=42)
    model.load_model(PATH)
    # Calculate scores
    model_score = model.score(X_valid, y_valid)
    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    mae = mean_absolute_error(y_valid, y_pred)

    # (neptune) Log scores
    run["training/val/r2"] = model_score
    run["training/val/rmse"] = rmse
    run["training/val/mae"] = mae

    # Visualize predictions
    df_result = pd.DataFrame(
        data={
            "y_valid": y_valid.values,
            "y_pred": y_pred,
            "Week": df_valid.loc[X_valid.index].Week,
        },
        index=X_valid.index,
    )
    df_result = df_result.set_index("Week")

    plt.figure()
    preds_plot = sns.lineplot(data=df_result)

    # (neptune) Log predictions visualizations
    run["training/plots/ypred_vs_y_valid"].upload(File.as_image(preds_plot.figure))
