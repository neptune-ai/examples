import argparse
import logging
import os

import neptune
import neptune.integrations.prophet as npt_utils
import pandas as pd
from prophet.serialize import model_from_json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_data", type=str, help="path to validation data")
    parser.add_argument("--neptune_project", type=str, help="neptune project to log to")
    parser.add_argument("--neptune_custom_run_id", type=str, help="neptune run id to log to")
    parser.add_argument("--neptune_api_token", type=str, help="neptune service account token")
    args = parser.parse_args()

    os.environ["NEPTUNE_PROJECT"] = args.neptune_project
    os.environ["NEPTUNE_CUSTOM_RUN_ID"] = args.neptune_custom_run_id
    os.environ["NEPTUNE_API_TOKEN"] = args.neptune_api_token
    valid_df = pd.read_csv(select_first_file(args.valid_data))

    def get_prophet_data_format(X, y):
        prophet_ds = X.copy()
        prophet_y = y.copy()
        return pd.DataFrame(
            {
                "ds": prophet_ds.Date.astype("datetime64[ns]"),
                "y": prophet_y.astype("float64"),
            }
        )

    # (neptune) Initialize Neptune run
    run = neptune.init_run(
        tags=["prophet", "walmart-sales"],
        name="Prophet",
    )

    # Get train data
    X_valid, y_valid = valid_df.drop(["Weekly_Sales"], axis=1), valid_df.Weekly_Sales

    future_prophet_data = get_prophet_data_format(X_valid, y_valid)

    run["model_version/serialized_model"].download()
    model_path = "serialized_model.json"
    with open(model_path, "r") as fin:
        model = model_from_json(fin.read())

    forecast = model.predict(future_prophet_data)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

    # Calculate scores
    r2 = r2_score(y_valid, forecast.yhat)
    rmse = mean_squared_error(y_valid, forecast.yhat, squared=False)
    mae = mean_absolute_error(y_valid, forecast.yhat)

    # (Neptune) Log scores
    run["training/val/r2"] = r2
    run["training/val/rmse"] = rmse
    run["training/val/mae"] = mae

    # Create predicitions visualizations
    forecast_fig = model.plot(forecast)
    forecast_components_fig = model.plot_components(forecast)

    # (Neptune) Log predictions visualizations
    run["forecast_plots/forecast"].upload(forecast_fig)
    run["forecast_plots/forecast_components"].upload(forecast_components_fig)

    # (Neptune) Initialize a Model and Model version
    from neptune.exceptions import NeptuneModelKeyAlreadyExistsError

    model_key = "PRO"
    project_key = run["sys/id"].fetch().split("-")[0]

    try:
        with neptune.init_model(key=model_key) as model:
            logging.info("Creating a new model version...")
            model_version = neptune.init_model_version(model=f"{project_key}-{model_key}")

    except NeptuneModelKeyAlreadyExistsError:
        logging.info(f"A model with the provided key {model_key} already exists in this project.")
        logging.info("Creating a new model version...")
        model_version = neptune.init_model_version(
            model=f"{project_key}-{model_key}", name="Prophet"
        )

    model_version.change_stage("staging")

    # (Neptune) Log model version details to run
    run["model_version/id"] = model_version["sys/id"].fetch()
    run["model_version/model_id"] = model_version["sys/model_id"].fetch()
    run["model_version/url"] = model_version.get_url()

    # (Neptune) Log run details to model version
    model_version["run/id"] = run["sys/id"].fetch()
    model_version["run/name"] = run["sys/name"].fetch()
    model_version["run/url"] = run.get_url()

    # (Neptune) Log model config to model registry
    model_version["config"] = run["model_config"].fetch()

    # (Neptune) Log scores to model version
    model_version["scores/r2"] = r2
    model_version["scores/rmse"] = rmse
    model_version["scores/mae"] = mae

    # (Neptune) Upload serialized model to model registry
    model_version["serialized_model"] = npt_utils.get_serialized_model(model)


if __name__ == "__main__":
    main()
