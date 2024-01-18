import argparse
import logging
import os

import pandas as pd

from utils import create_lags, encode_categorical_data, normalize_data


def data_preprocessing_component() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--train_data", type=str, help="path to train data")
    args = parser.parse_args()

    # # Load dataset
    df = pd.read_csv(args.data)
    logging.info(f"df loaded: {df}")

    # Normalize sales data
    df_normalized = normalize_data(df, "Weekly_Sales")

    # Encode categorical data
    df_encoded = df_normalized.copy()
    df_encoded = encode_categorical_data(df_encoded)

    # Create Lagged features
    df_encoded = create_lags(df_encoded)

    # Save train and validation data
    train_data_path = os.path.join(args.train_data, "train_data.csv")
    df_encoded.to_csv(train_data_path, index=False)

    logging.info(f"df encoded: {df_encoded}")


if __name__ == "__main__":
    data_preprocessing_component()
