import os
import sys
from pathlib import Path

import neptune
import pandas as pd
from azureml.core import Run
from neptune.integrations.xgboost import NeptuneCallback
from utils import *

# Get output path
mounted_output_path = sys.argv[2]
os.makedirs(mounted_output_path, exist_ok=True)
run = Run.get_context()

# # Load dataset
df = run.input_datasets["prepared_train_data"].to_pandas_dataframe()

# Drop index
df = df.drop("Column1", axis=1)

# Normalize sales data
df_normalized = normalize_data(df, "Weekly_Sales")

# Encode categorical data
df_encoded = df_normalized.copy()
df_encoded = encode_categorical_data(df_encoded)

# Create Lagged features
df_encoded = create_lags(df_encoded)

# Save train and validation data
TRAIN_DATA_PATH = os.path.join(mounted_output_path, "train_data.csv")
df_encoded.to_csv(TRAIN_DATA_PATH, index=False)
