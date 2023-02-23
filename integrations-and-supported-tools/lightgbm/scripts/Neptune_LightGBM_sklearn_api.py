import lightgbm as lgb
import neptune
from neptune.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Create run
run = neptune.init_run(
    project="common/lightgbm-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    name="sklearn-api-cls",
    tags=["lgbm-integration", "sklearn-api", "cls"],
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
    "num_leaves": 21,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 12,
    "n_estimators": 207,
}

# Create instance of the classifier object
gbm = lgb.LGBMClassifier(**params)

# Fit model and log metadata
gbm.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_names=["training", "validation"],
    eval_metric=["multi_logloss", "multi_error"],
    callbacks=[neptune_callback],
)

y_pred = gbm.predict(X_test)

# Log summary metadata to the same run under the "lgbm_summary" namespace
run["lgbm_summary"] = create_booster_summary(
    booster=gbm,
    log_trees=True,
    list_trees=[0, 1, 2, 3, 4],
    log_confusion_matrix=True,
    y_pred=y_pred,
    y_true=y_test,
)
