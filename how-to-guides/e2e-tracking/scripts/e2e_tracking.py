# ========================================== #
# How to track models end-to-end on Neptune? #
# ========================================== #
#
# This script shows how you can use Neptune to track a model across all stages of it's lifecycle by:
# * Logging model and run metadata to a central project
# * Comparing runs to select the best performing model
# * Monitoring a model once in production
#
# This script can be used as a template to design an automated end-to-end pipeline that covers the
# entire lifecycle of a model without needing any manual intervention.
#
# This example uses Optuna hyperparameter-optimization to simulate training and evaluating multiple
# scikit-learn models, and Evidently to monitor models in production.
# However, given Neptune's flexibility and multiple integrations, you can use any library and
# framework of your choice.
#
# List of all Neptune integrations: https://docs.neptune.ai/integrations/


# ===== Before you start ===== #
# This script example lets you try out Neptune anonymously, with zero setup.
# If you want to see the example logged to your own workspace instead:
# 1. Create a Neptune account --> https://neptune.ai/register
# 2. Create a Neptune project that you will use for tracking metadata.
#    Instructions --> https://docs.neptune.ai/setup/creating_project

## Import dependencies

import os
import pickle as pkl
import time

import matplotlib
import neptune
import numpy as np
import optuna
from evidently.metric_preset import RegressionPreset
from evidently.metrics import *
from evidently.report import Report
from neptune.exceptions import (
    NeptuneModelKeyAlreadyExistsError,
    ProjectNotFoundWithSuggestions,
)
from neptune.integrations.optuna import NeptuneCallback
from neptune.integrations.sklearn import create_regressor_summary, get_pickled_model
from neptune.utils import stringify_unsupported
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# To prevent `RuntimeError: main thread is not in main loop` error
matplotlib.use("Agg")

# ===== Track model training ===== #
# 1. Use Optuna to train multiple scikit-learn regression models
# 2. Leverage Neptune's scikit-learn and Optuna integrations to automatically log metadata and metrics
#    to Neptune for easy run comparison, while also using Neptune's model registry to track models.

## Prepare the dataset ##

data, target = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

### Create or initialize a Neptune project
# This step assumes that your Neptune API token is set as an environment variable
# **Note:** The "common/e2e-tracking" project used here is read-only.
# To log to your own project, just update `project_name`

# Comment the below 2 lines if your `NEPTUNE_PROJECT` env variable is already set
project_name = "common/e2e-tracking"
os.environ["NEPTUNE_PROJECT"] = project_name

try:
    # Initialize the project if it already exists
    project = neptune.init_project()
except ProjectNotFoundWithSuggestions:
    # Create the project if it does not already exist
    from neptune import management

    management.create_project(name=project_name)
    project = neptune.init_project()

### Create a new model in the model registry
# This model will serve as a placeholder for all the model versions created in different Optuna trials

model_key = "RFR"

try:
    # Create a new model if it does not already exist
    npt_model = neptune.init_model(key=model_key)
except NeptuneModelKeyAlreadyExistsError:
    # Initialize the model if it already exists
    npt_model = neptune.init_model(with_id=f"{project['sys/id'].fetch()}-{model_key}")

### Create the Optuna objective function
# We will create trial level runs and model versions within the objective function to capture
# trial-level metadata using Neptune's scikit-learn integration.


def objective(trial):

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 2, 64),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_samples_split": trial.suggest_int("min_samples_split", 3, 10),
    }

    # Create a trial-level run
    run_trial_level = neptune.init_run(
        capture_hardware_metrics=True,
        capture_stderr=True,
        capture_stdout=True,
        tags=["script", "trial-level"],
    )

    # Log study name and trial number to trial-level run
    run_trial_level["study-name"] = str(study.study_name)
    run_trial_level["trial-number"] = trial.number

    # Log parameters of a trial-level run
    run_trial_level["parameters"] = param

    # Train the model
    model = RandomForestRegressor(**param)
    model.fit(X_train, y_train)

    # Log model metadata to the trial level run
    run_trial_level["model_summary"] = create_regressor_summary(
        model, X_train, X_test, y_train, y_test
    )

    # Fetch objective score from the run
    run_trial_level.wait()
    score = run_trial_level["model_summary/test/scores/mean_absolute_error"].fetch()

    # Create a new model version
    model_version = neptune.init_model_version(model=f"{project['sys/id'].fetch()}-{model_key}")

    # Link model-version to the trial-level run
    model_version["training/run/id"] = run_trial_level["sys/id"].fetch()
    model_version["training/run/url"] = run_trial_level.get_url()

    run_trial_level["model_summary/id"] = model_version["sys/id"].fetch()
    run_trial_level["model_summary/url"] = model_version.get_url()

    # Log score to model version
    model_version["training/score"] = score

    # Upload model binary to model version
    model_version["saved_model"].upload(get_pickled_model(model))

    # Update model stage to "staging"
    model_version.change_stage("staging")

    # Stop model version and trial-level run
    model_version.stop()
    run_trial_level.stop()

    return score


### Create the Optuna study and a Neptune study-level run
# This run will have all the study-level metadata from Optuna,
# and can be used to group and compare runs across multiple HPO sweeps/studies

study = optuna.create_study(direction="minimize")

run_study_level = neptune.init_run(
    tags=["script", "study-level"],
    dependencies="infer",
)

run_study_level["study-name"] = study.study_name

### Initialize Neptune's Optuna callback
# This will log the HPO sweeps and trials to the study-level run
neptune_optuna_callback = NeptuneCallback(run_study_level)

### Run the hyperparameter-sweep with Neptune's Optuna callback
study.optimize(objective, n_trials=5, callbacks=[neptune_optuna_callback])

### Stop the study level run
run_study_level.stop()
print("Completed HPO")

# ===== Compare the runs, and choose the best model to move to production ===== #

### Download the model versions table as a pandas dataframe

model_versions_df = npt_model.fetch_model_versions_table(
    columns=["sys/stage", "training/score"],
    sort_by="training/score",
    ascending=True,
).to_pandas()

### Get scores and IDs of challenger and champion models
try:
    champion_model = model_versions_df[model_versions_df["sys/stage"] == "production"][
        "sys/id"
    ].values[0]
    champion_model_score = model_versions_df[model_versions_df["sys/stage"] == "production"][
        "training/score"
    ].values[0]
    print(f"Champion model ID: {champion_model} and score: {champion_model_score}")
    NO_CHAMPION = False
except IndexError:
    print("No model found in production")
    NO_CHAMPION = True

staged_models = model_versions_df[model_versions_df["sys/stage"] == "staging"]
challenger_model_score = min(staged_models["training/score"])
challenger_model_id = staged_models[staged_models["training/score"] == challenger_model_score][
    "sys/id"
].values[0]

print(f"Challenger model ID: {challenger_model_id} and score: {challenger_model_score}")

### Promote challenger to champion if score is better
if NO_CHAMPION:
    print(f"Promoting {challenger_model_id} to Production")
    with neptune.init_model_version(with_id=challenger_model_id) as challenger_model:
        challenger_model.change_stage("production")

elif challenger_model_score < champion_model_score:
    print("Challenger is better than champion")

    print(f"Archiving champion model {champion_model}")
    with neptune.init_model_version(with_id=champion_model) as champion_model:
        champion_model.change_stage("archived")

    print(f"Promoting {challenger_model_id} to Production")
    with neptune.init_model_version(with_id=challenger_model_id) as challenger_model:
        challenger_model.change_stage("production")

else:
    print("Champion model is better than challenger")
    print(f"Archiving challenger model {challenger_model_id}")
    with neptune.init_model_version(with_id=challenger_model_id) as challenger_model:
        challenger_model.change_stage("archived")

### Wait to sync with Neptune servers
time.sleep(5)

# ===== Monitor model in production ===== #
# In this section, you will:
# 1. Download the model binary from the model registry to make predictions in production
# 2. Use EvidentlyAI to monitor your model in production.
#
# You will use a modified version of the Evidently tutorial available here:
# https://docs.evidentlyai.com/get-started/tutorial.


### Setup
# You will use and example dataset and mock historical predictions to use as a reference.

data = fetch_california_housing(as_frame=True)
housing_data = data.frame

housing_data.rename(columns={"MedHouseVal": "target"}, inplace=True)

reference = housing_data.sample(n=10000, replace=False)
reference["prediction"] = reference["target"].values + np.random.normal(0, 5, reference.shape[0])

current = housing_data.sample(n=10000, replace=False)

### Download saved model from model registry
model_versions_df = npt_model.fetch_model_versions_table(columns=["sys/stage"]).to_pandas()

production_model = model_versions_df[model_versions_df["sys/stage"] == "production"][
    "sys/id"
].values[0]

npt_model_version = neptune.init_model_version(with_id=production_model)
npt_model_version["saved_model"].download()
print("Model binary downloaded to 'saved_model.pkl'")

### Make predictions from downloaded model on current test data
print("Making predictions on current data")

with open("saved_model.pkl", "rb") as f:
    model = pkl.load(f)

current["prediction"] = model.predict(current.drop(columns=["target"]))

### Generate regression report
reg_performance_report = Report(metrics=[RegressionPreset()])
reg_performance_report.run(reference_data=reference, current_data=current)

### Upload report to the model
reg_performance_report.save_html("report.html")
npt_model_version["production/report"].upload("report.html")
print(f"Model report uploaded to {npt_model_version.get_url()}")

### Upload metrics to the model
npt_model_version["production/metrics"] = stringify_unsupported(
    reg_performance_report.as_dict()["metrics"][0]
)
npt_model_version.wait()

### These metrics can then be fetched downstream to trigger a model refresh/retraining if needed
retraining_threshold = 0.5  # example

if (
    npt_model_version["production/metrics/result/current/mean_abs_error"].fetch()
    > retraining_threshold
):
    print("Model degradation detected. Retraining model...")
    ...
else:
    print("Model performance within expectations")

# ===== Stop Neptune objects ===== #
project.stop()
npt_model.stop()
npt_model_version.stop()
