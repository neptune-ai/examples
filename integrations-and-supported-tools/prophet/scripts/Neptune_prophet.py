import matplotlib
import neptune
import neptune.integrations.prophet as npt_utils
import pandas as pd
from prophet import Prophet

# To prevent `RuntimeError: main thread is not in main loop` error
matplotlib.use("Agg")

run = neptune.init_run(
    project="common/fbprophet-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["prophet", "script"],  # optional
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv"
)

# Market capacity
df["cap"] = 8.5


def nfl_sunday(ds) -> int:
    date = pd.to_datetime(ds)
    return 1 if date.weekday() == 6 and (date.month > 8 or date.month < 2) else 0


df["nfl_sunday"] = df.ds.apply(nfl_sunday)

model = Prophet()
model.add_regressor("nfl_sunday")
model.fit(df)

forecast = model.predict(df)

run["prophet_summary"] = npt_utils.create_summary(
    model=model,
    df=df,
    fcst=forecast,
    log_interactive=False,
)
