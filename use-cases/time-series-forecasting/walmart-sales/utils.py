import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def pre_process_data(df):
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


def normalize_data(df, column, n_std=2):
    print(f"Working on column: {column}")

    mean = df[column].mean()
    sd = df[column].std()

    df_norm = df[(df[column] <= mean + (n_std * sd))].copy()
    df_norm.loc[(df_norm[column] < 0), column] = 0  # remove negative numbers

    return df_norm


# ML
def create_lags(df: pd.DataFrame):
    lags = 9
    for i in range(1, lags):
        # Weekly Sales vs Lag-N
        df[f"Lag_{i}"] = df["Weekly_Sales"].shift(i)
        df[f"Lag_{i}"].fillna(df[f"Lag_{i}"].mean(), inplace=True)

    # Change sales column position to -1
    weekly_sales = df.pop("Weekly_Sales")
    df.insert(len(df.columns), "Weekly_Sales", weekly_sales)

    return df


def encode_categorical_data(df: pd.DataFrame):
    type_group = {"A": 1, "B": 2, "C": 3}
    df["Type"] = df["Type"].replace(type_group)  # changing A,B,C to 1-2-3
    df["IsHoliday"] = df["IsHoliday"].astype(bool).astype(int)
    df["Week"] = df["Week"].astype(int)

    return df


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


# DL
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": mean_absolute_error(df.value, df.prediction),
        "rmse": mean_squared_error(df.value, df.prediction) ** 0.5,
        "r2": r2_score(df.value, df.prediction),
    }
