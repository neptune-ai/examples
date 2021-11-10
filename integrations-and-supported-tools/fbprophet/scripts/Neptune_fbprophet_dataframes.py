import neptune.new as neptune
from fbprophet import Prophet
import pandas as pd



class NeptuneLogger:
    def __init__(self, run, base_namespace = 'experiment'):
        self.run = run
        self.base_namespace = base_namespace
        self.ns_run = run[self.base_namespace]
            
    def log_df(self, name, df: pd.DataFrame):
        if self.run.exists(f"{self.base_namespace}/dataframes/{name}"):
            raise ValueError(f"{name} already exists")
        else:
            self.ns_run[f"dataframes/{name}"].upload(neptune.types.File.as_html(df))



df = pd.read_csv("https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv")

df["cap"] = 8.5


# create a FBprophet model and set growth parameter to logistic
model = Prophet(growth='logistic')

# fit the model to the dataframe 
model.fit(df)

# show me the documentation for make_future_dataframe
model.make_future_dataframe(periods=365)

# make future dataframe with periods of 365 days * 5 years
future = model.make_future_dataframe(periods=365*5)
future["cap"] = 8.5


# Create a run
run = neptune.init(
    project='common/fbprophet-integration', 
    api_token="ANONYMOUS",
    tags=["fbprophet", "saturating forcasts", "script", "dataframes"]    
)


# log the future dataframe and df to neptune
npt_logger = NeptuneLogger(run)
npt_logger.log_df("future_df", future)
npt_logger.log_df("input_df", df)

# Stop the run
run.stop()