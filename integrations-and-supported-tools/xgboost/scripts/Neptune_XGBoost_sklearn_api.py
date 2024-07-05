# To fix the random RuntimeError: main thread is not in main loop error in Windows
import matplotlib.pyplot as plt
import neptune
import xgboost as xgb
from neptune.integrations.xgboost import NeptuneCallback
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

plt.switch_backend("agg")

# Create run
run = neptune.init_run(
    project="common/xgboost-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["xgb-integration", "sklearn-api", "script"],
)

# Create neptune callback
neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])

# Prepare data
data = fetch_california_housing()
y = data["target"]
X = data["data"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Define parameters
model_params = {
    "n_estimators": 70,
    "eta": 0.7,
    "gamma": 0.001,
    "max_depth": 9,
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
}

reg = xgb.XGBRegressor(**model_params, callbacks=[neptune_callback])

# Fit the model and log metadata to the run in Neptune
reg.fit(X_train, y_train)
