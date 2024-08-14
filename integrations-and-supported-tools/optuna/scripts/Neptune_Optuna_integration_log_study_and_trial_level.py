import os

import lightgbm as lgb
import neptune
import neptune.integrations.optuna as optuna_utils
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# To connect to the Neptune app, you need to tell Neptune who you are (`api_token`) and where to send the data (`project`).
# **By default, this script logs to the public project `common/optuna` as an anonymous user.**
# Note: Public projects are cleaned regularly, so anonymous runs are only stored temporarily.

# %%### Log to public project
os.environ["NEPTUNE_API_TOKEN"] = neptune.ANONYMOUS_API_TOKEN
os.environ["NEPTUNE_PROJECT"] = "common/optuna"

# **To Log to your own project instead**
# Uncomment the code block below:

# from getpass import getpass
# os.environ["NEPTUNE_API_TOKEN"]=getpass("Enter your Neptune API token: ")
# os.environ["NEPTUNE_PROJECT"]="workspace-name/project-name",  # replace with your own


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
    run_trial_level = neptune.init_run(tags=["trial", "script"])

    # log study name and trial number to trial-level run
    run_trial_level["sys/group_tags"].add([study.study_name])
    run_trial_level["trial/number"] = trial.number

    # log parameters of a trial-level run
    run_trial_level["trial/parameters"] = param

    # run model training
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(test_x)
    accuracy = roc_auc_score(test_y, preds)

    # log score of a trial-level run
    run_trial_level["trial/score"] = accuracy

    # stop trial-level run
    run_trial_level.stop()

    return accuracy


# create an Optuna study
study = optuna.create_study(direction="maximize")

# create a study-level run
run_study_level = neptune.init_run(tags=["study", "script"])

# add study name as a group tag to the study-level run
run_study_level["sys/group_tags"].add([study.study_name])

# create a study-level NeptuneCallback
neptune_callback = optuna_utils.NeptuneCallback(run_study_level)

# pass NeptuneCallback to the Study
study.optimize(objective_with_logging, n_trials=5, callbacks=[neptune_callback])

# stop study-level run
run_study_level.stop()

# Go to the Neptune app to filter and see all the runs for this run ID
