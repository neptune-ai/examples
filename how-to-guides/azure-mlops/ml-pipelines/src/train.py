import os
import sys

import joblib
import neptune
import pandas as pd
import xgboost as xgb
from azureml.core import Run
from neptune.integrations.xgboost import NeptuneCallback
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

mounted_output_path = sys.argv[2]
os.environ["NEPTUNE_PROJECT"] = sys.argv[3]
os.environ["NEPTUNE_CUSTOM_RUN_ID"] = sys.argv[4]
os.environ["NEPTUNE_API_TOKEN"] = neptune.ANONYMOUS_API_TOKEN

# dataset object from the run
azure_run = Run.get_context()
train_df = azure_run.input_datasets["prepared_train_data"].to_pandas_dataframe()

print(train_df)

# Split data into train and validation
features_to_exclude = ["Weekly_Sales", "Date", "Year"]
X = train_df.loc[:, ~train_df.columns.isin(features_to_exclude)]
y = train_df.loc[:, "Weekly_Sales"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=False
)

# Encoding data to categorical variables
lbl = preprocessing.OneHotEncoder(handle_unknown="ignore")
X_train = lbl.fit_transform(X_train)

# (neptune) Initialize Neptune run
run = neptune.init_run(
    tags=["MLOps", "baseline", "xgboost", "walmart-sales"],
    name="XGBoost",
)

neptune_callback = NeptuneCallback(run=run)

#  Train model
model = xgb.XGBRegressor(callbacks=[neptune_callback], random_state=42).fit(X_train, y_train)


# Save model
model_filename = "model.json"
model.save_model(model_filename)
run["training/model"].upload(model_filename)

# Save Label encoder
lbl_encoder_filename = "label_encoder.joblib"
joblib.dump(lbl, lbl_encoder_filename)
run["training/label_encoder"].upload(lbl_encoder_filename)

# Concatenate x and y train data
validation_df = pd.concat([X_val, y_val], axis=1)

# Save validation data
VALIDATION_DATA_PATH = os.path.join(mounted_output_path, "validation_data.csv")
validation_df.to_csv(VALIDATION_DATA_PATH, index=False)
