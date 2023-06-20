# To fix the random RuntimeError: main thread is not in main loop error in Windows running python 3.8
import matplotlib.pyplot as plt
import neptune
import neptune.integrations.sklearn as npt_utils
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

plt.switch_backend("agg")

run = neptune.init_run(
    project="common/sklearn-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    name="regression-example",
    tags=["RandomForestRegressor", "regression"],
)

parameters = {"n_estimators": 70, "max_depth": 7, "min_samples_split": 3}

rfr = RandomForestRegressor(**parameters)

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfr.fit(X_train, y_train)

run["rfr_summary"] = npt_utils.create_regressor_summary(rfr, X_train, X_test, y_train, y_test)
