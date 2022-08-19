import neptune.new as neptune
import neptune_prophet.impl as npt_utils
from prophet import Prophet
import pandas as pd

run = neptune.init(
    project='common/fbprophet-integration',
    api_token="ANONYMOUS",
    tags=["fbprophet","additional regressors", "script"], # optional
)

df = pd.read_csv("https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv")

df["cap"] = 8.5

def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0

df["nfl_sunday"] = df.ds.apply(nfl_sunday)

model = Prophet()
model.add_regressor("nfl_sunday")
model.fit(df)

forecast = model.predict(df)

run["prophet_summary"] = npt_utils.create_summary(model=model, df=df, fcst=forecast)