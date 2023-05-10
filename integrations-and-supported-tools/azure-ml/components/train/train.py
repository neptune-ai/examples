import argparse
import os

import neptune
import neptune.integrations.prophet as npt_utils
import pandas as pd
from neptune.utils import stringify_unsupported
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.model_selection import train_test_split


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
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--valid_data", type=str, help="path to validation data")
    parser.add_argument("--neptune_project", type=str, help="neptune project to log to")
    parser.add_argument("--neptune_custom_run_id", type=str, help="neptune run id to log to")
    parser.add_argument("--neptune_api_token", type=str, help="neptune service account token")
    args = parser.parse_args()

    os.environ["NEPTUNE_PROJECT"] = args.neptune_project
    os.environ["NEPTUNE_CUSTOM_RUN_ID"] = args.neptune_custom_run_id
    os.environ["NEPTUNE_API_TOKEN"] = args.neptune_api_token

    # (neptune) Initialize Neptune run
    run = neptune.init_run(
        tags=["prophet", "walmart-sales"],
        name="Prophet",
    )

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    def get_train_data(df: pd.DataFrame, features_to_exclude=None):
        if features_to_exclude is None:
            features_to_exclude = ["Weekly_Sales", "Date"]

        X = df.loc[:, ~df.columns.isin(features_to_exclude)]
        y = df.loc[:, "Weekly_Sales"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=False
        )
        return X_train, X_test, y_train, y_test

    def get_prophet_data_format(X, y):
        prophet_ds = X.copy()
        prophet_y = y.copy()
        return pd.DataFrame(
            {
                "ds": prophet_ds.Date.astype("datetime64[ns]"),
                "y": prophet_y.astype("float64"),
            }
        )

    X_train, X_valid, y_train, y_valid = get_train_data(
        train_df[train_df.Dept == 1], ["Weekly_Sales", "Year"]
    )

    prophet_data = get_prophet_data_format(X_train, y_train)

    # Train model
    model = Prophet(
        changepoint_range=0.8,
        seasonality_prior_scale=0.1,
        holidays_prior_scale=0.5,
        changepoint_prior_scale=0.1,
    )
    model.add_country_holidays(country_name="US")
    model.fit(prophet_data)

    run["model_config"] = stringify_unsupported(npt_utils.get_model_config(model))

    with open("serialized_model.json", "w") as fout:
        fout.write(model_to_json(model))
    run["model_version/serialized_model"].upload("serialized_model.json")

    # Concatenate x and y train data
    validation_df = pd.concat([X_valid, y_valid], axis=1)

    # Save train and validation data
    validation_data_path = os.path.join(args.valid_data, "validation_data.csv")
    validation_df.to_csv(validation_data_path, index=False)


if __name__ == "__main__":
    main()
