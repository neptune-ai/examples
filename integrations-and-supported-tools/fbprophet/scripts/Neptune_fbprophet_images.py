import neptune.new as neptune
from fbprophet import Prophet
import itertools
import pandas as pd
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt


class NeptuneLogger:
    def __init__(self, run, base_namespace = 'experiment'):
        self.run = run
        self.base_namespace = base_namespace
        self.ns_run = run[self.base_namespace]
        
    def log_images(self, name, fig, path=None):
        # using neptune new api use exists method to check if the file already exists 
        if self.run.exists(f"{self.base_namespace}/images/{name}"):
            # raise an exception because name already exists
            raise ValueError(f"{name} already exists")
        else:
            if path:
                self.ns_run[f"images/{name}"].upload(path)
            else:
                if isinstance(fig, list):
                    # if the last object is a plt.Line2D object then use neptune new api to log it using upload method and wrap the figure with neptune.types.File.as_image
                    if isinstance(fig[-1], plt.Line2D):
                        self.ns_run[f"images/{name}"].upload(neptune.types.File.as_image(fig[-1].figure))
                else:
                    self.ns_run[f"images/{name}"].upload(neptune.types.File.as_image(fig))
            

 


df = pd.read_csv("https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_R.csv")

df["cap"] = 8.5


# create a FBprophet model and set growth parameter to logistic
model = Prophet(growth='logistic')
model.fit(df)
model.make_future_dataframe(periods=365)

future = model.make_future_dataframe(periods=365*5)
future["cap"] = 8.5

forecast = model.predict(future)

# plot the predictions
fig = model.plot(forecast)

# Create a new run
run = neptune.init(
    project='common/fbprophet-integration', 
    api_token="ANONYMOUS",
    tags=["fbprophet", "saturating forcasts", "script", "images"]    
)


# log the model, fig, params, future dataframe and df to neptune
npt_logger = NeptuneLogger(run)
npt_logger.log_images("forcast",fig)

# Stop the run
run.stop()