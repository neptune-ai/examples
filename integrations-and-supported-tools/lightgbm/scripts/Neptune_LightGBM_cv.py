import lightgbm as lgb
import neptune
from neptune.integrations.lightgbm import NeptuneCallback
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Create run
run = neptune.init_run(
    project="common/lightgbm-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    name="cv-cls",
    tags=["lgbm-integration", "cv", "cls"],
)

# Create neptune callback
neptune_callback = NeptuneCallback(run=run)

# Prepare data
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Define parameters
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 10,
    "metric": ["multi_logloss", "multi_error"],
    "num_leaves": 21,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 12,
}

# Run CV
gbm_cv = lgb.cv(
    params,
    lgb_train,
    num_boost_round=200,
    nfold=7,
    callbacks=[neptune_callback],
)
