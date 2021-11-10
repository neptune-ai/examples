import neptune.new as neptune
from fbprophet import Prophet
import itertools
import pandas as pd
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


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
    

    def log_images(self, name, fig, path=None):
        if self.run.exists(f"{self.base_namespace}/images/{name}"):
            raise ValueError(f"{name} already exists")
        else:
            if path:
                self.ns_run[f"images/{name}"].upload(path)
            else:
                if isinstance(fig, list):
                    if isinstance(fig[-1], plt.Line2D):
                        self.ns_run[f"images/{name}"].upload(neptune.types.File.as_image(fig[-1].figure))
                else:
                    self.ns_run[f"images/{name}"].upload(neptune.types.File.as_image(fig))
            

    def log_df(self, name, df: pd.DataFrame):
        if self.run.exists(f"{self.base_namespace}/dataframes/{name}"):
            raise ValueError(f"{name} already exists")
        else:
            self.ns_run[f"dataframes/{name}"].upload(neptune.types.File.as_html(df))


    def log_artifacts(self, name, path):
        self.ns_run[f"artifacts/{name}"].track_files(path)



dataset_path = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_air_passengers.csv"
# Use case 4 - Hyperparameter tuning and cross validation.
df = pd.read_csv(dataset_path)
df.head()

# create a neptune run
run = neptune.init(
    project="common/fbprophet-integration",
    api_token="ANONYMOUS",
    tags=["fbprophet", "cross validation", "script", "hpo"]
) 


param_grid = {  
    'changepoint_prior_scale': [0.009, 0.01]
}
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = [] 


# Use cross validation to evaluate all parameters
for i, params in enumerate(all_params):
    m = Prophet(**params).fit(df)  # Fit model with given params
    df_cv = cross_validation(m, initial='30 days', horizon='30 days', parallel="threads")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

    npt_logger = NeptuneLogger(run, base_namespace = f"experiment/cv_{i}")

    npt_logger.log_config(m)
    npt_logger.log_df("df", df)
    npt_logger.log_df("df_cv", df_cv)
    npt_logger.log_df("df_p", df_p)


fig = plot_cross_validation_metric(df_cv, metric='mape')

# log fig to neptune
npt_logger = NeptuneLogger(run)
npt_logger.log_images("cross validation", fig)

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses

# log tuning results to neptune
npt_logger.log_df("tuning_results", tuning_results)

# get the best parameters using all_params, numpy and rmses
best_params = all_params[np.argmin(rmses)]

#log best_params to neptune
npt_logger.ns_run["best_params"] = best_params

# stop neptune session
run.stop()