import sys

import matplotlib.pyplot as plt
import neptune
import pandas as pd
import seaborn as sns
import xgboost as xgb
from neptune.integrations.xgboost import NeptuneCallback
from neptune.types import File
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append("../")
from utils import *

sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"
plt.ioff()

# (neptune) Initialize Neptune run
run = neptune.init_run(
    tags=["baseline", "xgboost", "walmart-sales"],
    name="XGBoost",
)
neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])

# Load dataset
DATA_PATH = "../dataset"
df = load_data(DATA_PATH)

# Normalize sales data
df_normalized = normalize_data(df, "Weekly_Sales")

# Encode categorical data
df_encoded = df_normalized.copy()
df_encoded = encode_categorical_data(df_encoded)

# Create Lagged features
df_encoded = create_lags(df_encoded)

# Get train data
X_train, X_valid, y_train, y_valid = get_train_data(
    df_encoded[df_encoded.Dept == 1], ["Weekly_Sales", "Date", "Year"]
)

# Train model
model = xgb.XGBRegressor(callbacks=[neptune_callback]).fit(
    X_train,
    y_train,
)

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
        "Week": df_encoded.loc[X_valid.index].Week,
    },
    index=X_valid.index,
)
df_result = df_result.set_index("Week")

plt.figure()
preds_plot = sns.lineplot(data=df_result)

# (neptune) Log predictions visualizations
run["training/plots/ypred_vs_y_valid"].upload(File.as_image(preds_plot.figure))
