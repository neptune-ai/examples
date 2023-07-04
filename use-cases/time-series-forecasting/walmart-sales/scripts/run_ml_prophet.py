import sys

import matplotlib.pyplot as plt
import neptune
import neptune.integrations.prophet as npt_utils
import seaborn as sns
from neptune.types import File
from neptune.utils import stringify_unsupported
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append("../")
from utils import *

sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"
plt.ioff()


# (Neptune) Initialize Neptune run
run = neptune.init_run(
    tags=["prophet", "walmart-sales"],
    name="Prophet",
)

# Load dataset
DATA_PATH = "../dataset"
df = load_data(DATA_PATH, cache=True)

# Normalize sales data
df_normalized = normalize_data(df, "Weekly_Sales")

# Encode categorical data
df_encoded = df_normalized.copy()
df_encoded = encode_categorical_data(df_encoded)

# Create Lagged features
df_encoded = create_lags(df_encoded)

# Get train data
X_train, X_valid, y_train, y_valid = get_train_data(
    df_encoded[df_encoded.Dept == 1], ["Weekly_Sales", "Year"]
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

future_prophet_data = get_prophet_data_format(X_valid, y_valid)
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
        print("Creating a new model version...")
        model_version = neptune.init_model_version(model=f"{project_key}-{model_key}")

except NeptuneModelKeyAlreadyExistsError:
    print(f"A model with the provided key {model_key} already exists in this project.")
    print("Creating a new model version...")
    model_version = neptune.init_model_version(model=f"{project_key}-{model_key}", name="Prophet")

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

# (Neptune) Compare challenger model with champion
## (Neptune) Fetch score of current champion model
with neptune.init_model(with_id=f"{project_key}-{model_key}") as model:
    model_versions_df = model.fetch_model_versions_table().to_pandas()

champion_score = model_versions_df.loc[model_versions_df["sys/stage"] == "production"][
    "scores/rmse"
].values[0]

## (Neptune) Promote challenger to champion if it outperforms

print(f"Champion model score: {champion_score}\nChallenger model score: {rmse}")

if rmse < champion_score:
    print("Archiving champion model")
    champion_model = model_versions_df.loc[model_versions_df["sys/stage"] == "production"][
        "sys/id"
    ].values[0]
    with neptune.init_model_version(with_id=champion_model) as npt_prod_model:
        npt_prod_model.change_stage("archived")

    print("Promoting challenger to champion")
    model_version.change_stage("production")

else:
    print("Archiving challenger model")
    model_version.change_stage("archived")

# (Neptune) Stop Neptune objects
