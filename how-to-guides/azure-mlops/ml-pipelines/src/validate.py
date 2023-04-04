import os
import sys

import joblib
import neptune
import pandas as pd
import seaborn as sns
import xgboost as xgb
from azureml.core import Run
from matplotlib import pyplot as plt
from neptune.types import File
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["NEPTUNE_PROJECT"] = sys.argv[2]
os.environ["NEPTUNE_CUSTOM_RUN_ID"] = sys.argv[3]
os.environ["NEPTUNE_API_TOKEN"] = neptune.ANONYMOUS_API_TOKEN

# Load data
azure_run = Run.get_context()
valid_df = azure_run.input_datasets["prepared_val_data"].to_pandas_dataframe().copy()

# Get train data
X_valid, y_valid = valid_df.drop(["Weekly_Sales"], axis=1), valid_df.Weekly_Sales

# (neptune) Initialize Neptune run
run = neptune.init_run(
    tags=["MLOps", "baseline", "xgboost", "walmart-sales"],
    name="XGBoost",
)

# Load model checkpoint from model registry
run["training/model"].download()
model_path = "model.json"
model = xgb.XGBRegressor(random_state=42)
model.load_model(model_path)

# Load label encoder
run["training/label_encoder"].download()
lbl_encoder_path = "label_encoder.joblib"
lbl = joblib.load(lbl_encoder_path)
X_valid = lbl.transform(X_valid)

# Calculate scores
model_score = model.score(X_valid, y_valid)
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
mae = mean_absolute_error(y_valid, y_pred)

# # # (neptune) Log scores
run["training/val/r2"] = model_score
run["training/val/rmse"] = rmse
run["training/val/mae"] = mae

# Visualize predictions
sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"
plt.ioff()

df_result = pd.DataFrame(
    data={
        "y_valid": y_valid.values,
        "y_pred": y_pred,
        "Week": valid_df.loc[valid_df.index].Week,
    },
    index=valid_df.index,
)
df_result = df_result.set_index("Week")

plt.figure()
preds_plot = sns.lineplot(data=df_result)

# (neptune) Log predictions visualizations
run["training/plots/ypred_vs_y_valid"].upload(File.as_image(preds_plot.figure))
