import neptune.new as neptune
from fbprophet import Prophet
import pandas as pd
import numpy as np
import sys
import copy


class NeptuneLogger:
    def __init__(self, run, base_namespace = 'experiment'):
        self.run = run
        self.base_namespace = base_namespace
        self.ns_run = run[self.base_namespace]
        
    def log_config(self, model: Prophet):
        module = "numpy"
        if module not in sys.modules:
            raise Exception(f"{module} is not imported")
            
        config = copy.deepcopy(model.__dict__)
        
        model.history_dates = pd.DataFrame(model.history_dates)

        with open("trend.npy", 'wb') as f:
            np.save(f, config["params"]["trend"])

        config["params"].pop("trend")
            
     
        self.ns_run["config/params/trend"].upload("./trend.npy")

        for key, value in config.items():
            if isinstance(value, pd.DataFrame):
                self.ns_run[f"config/{key}"].upload(neptune.types.File.as_html(value))
            elif isinstance(value, np.ndarray):
                self.ns_run[f"config/{key}"].upload(neptune.types.File.as_html(pd.DataFrame(value)))
            elif isinstance(value, pd.Series):
                self.ns_run[f"config/{key}"].upload(neptune.types.File.as_html(pd.DataFrame(value)))
            else:
                self.ns_run[f"config/{key}"] = value


df = pd.read_csv("https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv")

df["cap"] = 8.5


# create a FBprophet model and set growth parameter to logistic
model = Prophet(growth='logistic')
model.fit(df)
model.make_future_dataframe(periods=365)

future = model.make_future_dataframe(periods=365*5)
future["cap"] = 8.5

forecast = model.predict(future)
fig = model.plot(forecast)

# Create the run
run = neptune.init(
    project='common/fbprophet-integration', 
    api_token="ANONYMOUS",
    tags=["fbprophet", "saturating forcasts", "script", "model_config"]    
)


# log the model, fig, params, future dataframe and df to neptune
npt_logger = NeptuneLogger(run)
npt_logger.log_config(model)

# stop the run
run.stop()