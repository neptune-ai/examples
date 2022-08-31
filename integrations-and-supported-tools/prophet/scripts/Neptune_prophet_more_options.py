import neptune.new as neptune
import neptune.new.integrations.prophet as npt_utils
import pandas as pd
from prophet import Prophet

run = neptune.init_run(
    project="common/fbprophet-integration",
    api_token="ANONYMOUS",
    tags=["fbprophet", "additional regressors", "script"],  # optional
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv"
)

# Market capacity
df["cap"] = 8.5


def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    return 1 if date.weekday() == 6 and (date.month > 8 or date.month < 2) else 0


df["nfl_sunday"] = df.ds.apply(nfl_sunday)

model = Prophet()
model.add_regressor("nfl_sunday")
model.fit(df)

forecast = model.predict(df)

# Log Prophet plots to Neptune
run["forecast_plots"] = npt_utils.create_forecast_plots(model, forecast)
run["forecast_components"] = npt_utils.get_forecast_components(model, forecast)
run["residual_diagnostics_plot"] = npt_utils.create_residual_diagnostics_plots(forecast, df.y)

# Log Prophet model configuration
run["model_config"] = npt_utils.get_model_config(model)

# Log Prophet serialized model
run["model"] = npt_utils.get_serialized_model(model)
