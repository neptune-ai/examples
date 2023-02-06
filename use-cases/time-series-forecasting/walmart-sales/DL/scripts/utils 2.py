import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def pre_process_data(df: pd.DataFrame):

    # process dates and create year, month and week features
    df["Date"] = pd.to_datetime(df.Date)
    df["Year"] = pd.DatetimeIndex(df.Date).year
    df["Month"] = pd.DatetimeIndex(df.Date).month
    df["Week"] = df.Date.dt.isocalendar().week

    # convert from F to C
    df["Temperature"] = pd.DataFrame((df.Temperature.values - 32) * 5 / 9)

    # fill missing values
    df["MarkDown1"].fillna(df["MarkDown1"].mean(), inplace=True)
    df["MarkDown2"].fillna(df["MarkDown2"].mean(), inplace=True)
    df["MarkDown3"].fillna(df["MarkDown3"].mean(), inplace=True)
    df["MarkDown4"].fillna(df["MarkDown4"].mean(), inplace=True)
    df["MarkDown5"].fillna(df["MarkDown5"].mean(), inplace=True)

    # change position of weekly sales column to last
    weekly_sales = df.pop("Weekly_Sales")
    df.insert(len(df.columns), "Weekly_Sales", weekly_sales)
    return df


def load_data(path, cache=False, all_df=False):
    if os.path.exists("aggregate_data.csv") and cache == True:
        return pd.read_csv(f"{path}/aggregate_data.csv", index_col=0)

    df_train = pd.read_csv(f"{path}/train.csv")
    df_fts = pd.read_csv(f"{path}/features.csv")
    df_stores = pd.read_csv(f"{path}/stores.csv")
    df = pd.merge(df_train, df_stores)
    df = pd.merge(df, df_fts)
    df = pre_process_data(df)
    df.to_csv(f"{path}/aggregate_data.csv")

    return (df, df_train, df_fts, df_stores) if all_df else df


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


def calculate_metrics(df):
    return {
        "mae": mean_absolute_error(df.value, df.prediction),
        "rmse": mean_squared_error(df.value, df.prediction) ** 0.5,
        "r2": r2_score(df.value, df.prediction),
    }


def get_model_ckpt_name(run):
    return list(run.get_structure()["training"]["model"]["checkpoints"].keys())[-1]
