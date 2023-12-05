import lightgbm as lgb
import neptune
import neptune.integrations.optuna as optuna_utils
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def objective(trial):
    data, target = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "verbose": -1,
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0, step=0.1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0, step=0.1),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(test_x)
    return roc_auc_score(test_y, preds)


# Fetch an existing Neptune run where you logged the Optuna Study
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/optuna-integration",
    with_id="NEP1-18517",
    monitoring_namespace="monitoring",
)  # you can pass your credentials and run ID here

# Load the Optuna Study from Neptune run
study = optuna_utils.load_study_from_run(run)

# Continue logging to the existing Neptune run
neptune_callback = optuna_utils.NeptuneCallback(run)
study.optimize(objective, n_trials=2, callbacks=[neptune_callback])

# Stop logging to a Neptune run
run.stop()
