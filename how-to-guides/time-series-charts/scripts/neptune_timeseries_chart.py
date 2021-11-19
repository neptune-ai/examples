import neptune.new as neptune
import pandas as pd

df = pd.read_csv("../data/neptue_timeseries_toupload_example.csv")

values = df['VALUE'].values.tolist()

# convert datemtime to unix timestamp seconds
dates = (pd.to_datetime(df['DATE']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

run = neptune.init(
    project= 'common/showroom',
    api_token='ANONYMOUS'
)

for v, d in zip(values, dates):
    run['timeseries'].log(value= v, timestamp=int(d))

run.stop()