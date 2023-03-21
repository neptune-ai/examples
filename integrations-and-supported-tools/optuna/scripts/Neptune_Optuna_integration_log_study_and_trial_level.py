import uuid

import lightgbm as lgb
import neptune
import neptune.integrations.optuna as optuna_utils
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# create a sweep ID
sweep_id = uuid.uuid1()
print("sweep-id: ", sweep_id)

# create a study-level run
run_study_level = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN, project="common/optuna-integration"
)

# pass the sweep ID to study-level run
run_study_level["sys/tags"].add("study-level")
run_study_level["sweep-id"] = str(sweep_id)


# create an objective function that logs each trial as a separate Neptune run
def objective_with_logging(trial):
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

    # create a trial-level run
    run_trial_level = neptune.init_run(
        api_token=neptune.ANONYMOUS_API_TOKEN, project="common/optuna-integration"
    )

    # log sweep id to trial-level run
    run_trial_level["sys/tags"].add("trial-level")
    run_trial_level["sweep-id"] = str(sweep_id)

    # log parameters of a trial-level run
    run_trial_level["parameters"] = param

    # run model training
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(test_x)
    accuracy = roc_auc_score(test_y, preds)

    # log score of a trial-level run
    run_trial_level["score"] = accuracy

    # stop trial-level run
    run_trial_level.stop()

    return accuracy


# create a study-level NeptuneCallback
neptune_callback = optuna_utils.NeptuneCallback(run_study_level)

# pass NeptuneCallback to the Study
study = optuna.create_study(direction="maximize")
study.optimize(objective_with_logging, n_trials=5, callbacks=[neptune_callback])

# stop study-level run
run_study_level.stop()

# Go to the Neptune app to filter and see all the runs for this run ID
