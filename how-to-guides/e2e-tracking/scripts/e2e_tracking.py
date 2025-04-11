# %% ========================================== #
#     How to track models end-to-end in Neptune #
# ============================================= #
#
# This script shows how you can use Neptune to track a model across all stages of its lifecycle by:
# * Logging model and run metadata to a central project
# * Grouping models by their stage
# * Comparing models to select the best performing model
# * Monitoring a model once in production
#
# This script can be used as a template to design an automated end-to-end pipeline that covers the
# entire lifecycle of a model without needing any manual intervention.
#
# This example uses Optuna hyperparameter-optimization to simulate training and evaluating multiple
# XGBoost models, and Evidently to monitor models in production.
# However, given Neptune's flexibility and multiple integrations, you can use any library and
# framework of your choice.
#
# List of all Neptune integrations: https://docs-legacy.neptune.ai/integrations/

# ===== Before you start ===== #
# This script example lets you try out Neptune anonymously, with zero setup.
# If you want to see the example logged to your own workspace instead:
# 1. Create a Neptune account --> https://neptune.ai/register
# 2. Create a Neptune project that you will use for tracking metadata.
#    Instructions --> https://docs-legacy.neptune.aitune.ai/setup/creating_project

# %%# Import dependencies

import os
import pickle as pkl

import matplotlib
import neptune
import numpy as np
import optuna
import xgboost as xgb
from evidently.metric_preset import RegressionPreset
from evidently.metrics import *
from evidently.report import Report
from neptune.integrations.optuna import NeptuneCallback
from neptune.integrations.xgboost import NeptuneCallback as XGBNeptuneCallback
from neptune.utils import stringify_unsupported
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# To prevent `RuntimeError: main thread is not in main loop` error
matplotlib.use("Agg")

# %% ===== Track model training ===== #
# 1. Use Optuna to train multiple XGBoost models
# 2. Leverage Neptune's XGBoost and Optuna integrations to automatically log metadata and metrics
#    to Neptune for easy run comparison

# %%## Initialize the Neptune project
# To connect to the Neptune app, you need to tell Neptune who you are (`api_token`) and where to send the data (`project`).
# **By default, this script logs to the public project `common/e2e-tracking` as an anonymous user.**
# Note: Public projects are cleaned regularly, so anonymous runs are only stored temporarily.

# %%### Log to public project
os.environ["NEPTUNE_API_TOKEN"] = neptune.ANONYMOUS_API_TOKEN
os.environ["NEPTUNE_PROJECT"] = "common/e2e-tracking"

# %%### **To Log to your own project instead**
# Uncomment the code block below:

# from getpass import getpass
# os.environ["NEPTUNE_API_TOKEN"]=getpass("Enter your Neptune API token: ")
# os.environ["NEPTUNE_PROJECT"]="workspace-name/project-name",  # replace with your own

project = neptune.init_project(mode="read-only")

# %%# Prepare the dataset ##

data, target = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)

evals = [(dtrain, "train"), (dval, "valid")]

# %%## Create the Optuna objective function
# We will create trial level runs and model versions within the objective function to capture
# trial-level metadata using Neptune's XGBoost integration.


def objective(trial):
    # Define model parameters
    model_params = {
        "max_depth": trial.suggest_int("max_depth", 0, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.75),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "objective": "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
    }

    # Define training parameters
    train_params = {
        "num_boost_round": trial.suggest_int("num_boost_round", 10, 50),
    }

    # Create a trial-level run
    run_trial_level = neptune.init_run(tags=["script", "trial-level"])

    # Log study name and trial number to trial-level run
    run_trial_level["study-name"] = str(study.study_name)
    run_trial_level["trial-number"] = trial.number

    # Log training parameters of a trial-level run
    run_trial_level["training/parameters"] = train_params

    # Model parameters are logged automatically by the NeptuneCallback

    # Create NeptuneCallback to log trial-level metadata
    neptune_xgb_callback = XGBNeptuneCallback(run=run_trial_level)

    # Train the model and log trial-level metadata to the trial-level run
    model = xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=train_params["num_boost_round"],
        evals=evals,
        verbose_eval=False,
        callbacks=[
            neptune_xgb_callback,
            xgb.callback.LearningRateScheduler(lambda epoch: 0.99**epoch),
            xgb.callback.EarlyStopping(rounds=10, save_best=True, maximize=False),
        ],
    )

    # Use group tags to identify the stage of the model
    run_trial_level["sys/group_tags"].add(["development"])

    # Stop trial-level run
    run_trial_level.stop()

    return model.best_score


# %%## Create the Optuna study and a Neptune study-level run
# This run will have all the study-level metadata from Optuna,
# and can be used to group and compare runs across multiple HPO sweeps/studies

study = optuna.create_study(direction="minimize")

run_study_level = neptune.init_run(
    tags=["script", "study-level"],
    dependencies="infer",
)

run_study_level["study-name"] = study.study_name

# %%## Initialize Neptune's Optuna callback
# This will log the HPO sweeps and trials to the study-level run
neptune_optuna_callback = NeptuneCallback(run_study_level)

# %%## Run the hyperparameter-sweep with Neptune's Optuna callback
study.optimize(objective, n_trials=5, callbacks=[neptune_optuna_callback])

# %%## Stop the study level run
run_study_level.stop()
print("Completed Study")

# %% ===== Compare the runs, and choose the best model to move to production ===== #

### Download the runs table as a pandas dataframe

score_namespace = (
    "training/early_stopping/best_score"  # This is where the best score is logged in Neptune
)

project = neptune.init_project(mode="read-only")

# %% ### Fetch the champion model. This is the model currently in production
champion_model_df = project.fetch_runs_table(
    query='`sys/group_tags`:stringSet CONTAINS "production"',
    columns=[score_namespace],
    sort_by=score_namespace,
    ascending=True,
    limit=1,
).to_pandas()

# %% ### Fetch the challenger model. This is the best model in development
challenger_model_df = project.fetch_runs_table(
    query='(`sys/group_tags`:stringSet CONTAINS "development") AND (`sys/tags`:stringSet CONTAINS "trial-level")',
    columns=[score_namespace],
    sort_by=score_namespace,
    ascending=True,
    limit=1,
).to_pandas()

# %%## Get scores and IDs of challenger and champion models
try:
    champion_model_id = champion_model_df["sys/id"].values[0]
    champion_model_score = champion_model_df[score_namespace].values[0]
    print(f"Champion model ID: {champion_model_id} and score: {champion_model_score}")
    NO_CHAMPION = False
except KeyError:
    print("❌ No model found in production")
    NO_CHAMPION = True

challenger_model_id = challenger_model_df["sys/id"].values[0]
challenger_model_score = challenger_model_df[score_namespace].values[0]

print(f"Challenger model ID: {challenger_model_id} and score: {challenger_model_score}")

# %%## Promote challenger to champion if score is better
if NO_CHAMPION:
    print(f"Promoting {challenger_model_id} to Production")
    with neptune.init_run(with_id=challenger_model_id) as challenger_model:
        challenger_model["sys/group_tags"].add("production")
        challenger_model["sys/group_tags"].remove("development")

elif challenger_model_score < champion_model_score:
    print("Challenger is better than champion")

    print(f"Archiving champion model {champion_model_id}")
    with neptune.init_run(with_id=champion_model_id) as champion_model:
        champion_model["sys/group_tags"].remove("production")
        champion_model["sys/group_tags"].add("archived")

    print(f"Promoting {challenger_model_id} to Production")
    with neptune.init_run(with_id=challenger_model_id) as challenger_model:
        challenger_model["sys/group_tags"].add("production")
        challenger_model["sys/group_tags"].remove("development")

else:
    print("Champion model is better than challenger")
    print(f"Archiving challenger model {challenger_model_id}")
    with neptune.init_run(with_id=challenger_model_id) as challenger_model:
        challenger_model["sys/group_tags"].add("archived")
        challenger_model["sys/group_tags"].remove("development")

# %% ===== Monitor model in production ===== #
# In this section, you will:
# 1. Download the model binary from the run to make predictions in production
# 2. Use EvidentlyAI to monitor your model in production.
#
# We will use a modified version of a tutorial from the Evidently documentation:
# https://docs.evidentlyai.com/get-started/tutorial

# %%## Setup
# You will use and example dataset and mock historical predictions to use as a reference.

data = fetch_california_housing(as_frame=True)
housing_data = data.frame

housing_data.rename(columns={"MedHouseVal": "target"}, inplace=True)

reference = housing_data.sample(n=10000, replace=False)
reference["prediction"] = reference["target"].values + np.random.normal(
    0, 0.1, reference.shape[0]
)  # Mocking historical predictions

current = housing_data.sample(n=10000, replace=False)
dcurrent = xgb.DMatrix(current.drop("target", axis=1), label=current["target"])

# %%## Download saved model from Neptune
production_model_id = (
    project.fetch_runs_table(
        query='`sys/group_tags`:stringSet CONTAINS "production"',
        columns=[],
        sort_by=score_namespace,
        ascending=True,
        limit=1,
    )
    .to_pandas()["sys/id"]
    .values[0]
)

production_model = neptune.init_run(with_id=production_model_id)
production_model["training/pickled_model"].download()

# %%## Make predictions from downloaded model on current test data
print("Making predictions on current data")

with open("pickled_model.pkl", "rb") as f:
    model = pkl.load(f)

current["prediction"] = model.predict(dcurrent)

# %%## Generate regression report
reg_performance_report = Report(metrics=[RegressionPreset()])
reg_performance_report.run(reference_data=reference, current_data=current)

# %%## Upload report to the model
reg_performance_report.save_html("report.html")
production_model["production/report"].upload("report.html")
print(f"Model report uploaded to {production_model.get_url()}")

# %%## Upload metrics to the model
production_model["production/metrics"] = stringify_unsupported(
    reg_performance_report.as_dict()["metrics"][0]
)
production_model.wait()

# %%## These metrics can then be fetched downstream to trigger a model refresh or retraining, if needed
retraining_threshold = 0.5  # example threshold

if production_model["production/metrics/result/current/rmse"].fetch() > retraining_threshold:
    print("Model degradation detected. Retraining model...")
    ...
else:
    print("Model performance within expectations")

# %% ===== Stop Neptune objects ===== #
project.stop()
production_model.stop()
