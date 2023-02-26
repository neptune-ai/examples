#!/usr/bin/env python
# coding: utf-8

# # Text classification using fastText and Optuna with Neptune tracking

# ## (Neptune) Install the neptune-notebooks widget
# The neptune-notebooks jupyter extension lets you version, manage and share notebook checkpoints in your projects, without leaving your notebook.
# [Read the docs](https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/jupyter-lab-and-jupyter-notebook)

# ## Setup

# ### Import dependencies

import csv
import os
import re
from io import StringIO
from pathlib import Path

import fasttext
import nltk
import optuna
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import stopwords
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

nltk.download("stopwords")
pd.options.plotting.backend = "plotly"
path = Path()

# ### Set parameters

N_JOBS = 2
N_TRIALS = 1
UPLOAD_SIZE_THRESHOLD = 300  # in MB

# ### Load input files

DATASET_PATH_S3 = "s3://neptune-examples/data/text-classification"

df_raw = pd.read_csv(f"{DATASET_PATH_S3}/legal_text_classification.csv")
df_raw.dropna(subset=["case_text"], inplace=True)
df_raw.drop_duplicates(subset="case_text", inplace=True)

# ## (Neptune) Initialize a neptune project
# A project is a collection of runs, models, and other metadata created by project members. Typically you should create one project per machine learning task, to make it easy to compare runs that are connected to building certain kinds of ML model.
# [Read the docs](https://docs.neptune.ai/you-should-know/core-concepts#project)

import neptune.new as neptune

WORKSPACE_NAME = "showcase"
PROJECT_NAME = "project-text-classification"

project = neptune.init_project(project=f"{WORKSPACE_NAME}/{PROJECT_NAME}")

# ## (Neptune) Log project level metadata
# All metadata common across all runs in a project (for example - input and configuration files) should be logged at the project level itself for easier management

# ### (Neptune) Version and track datasets
# Neptune lets you track pinters to datasets, models, and other artifacts stored locally or in S3.
# [Read the docs](https://docs.neptune.ai/how-to-guides/data-versioning)

project["fasttext/data/files"].track_files(DATASET_PATH_S3)


# ### (Neptune) Log dataset sample
# Smaller artifacts can also be uploaded directly to Neptune.
# [Read the docs](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#files)

from neptune.new.types import File

csv_buffer = StringIO()
df_raw.sample(100).to_csv(csv_buffer, index=False)
project["fasttext/data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))

# ### (Neptune) Log metadata plots
# Similar to other artifacts, you can also upload images and plot objects to Neptune.
# [Read the docs](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#images)

fig = df_raw.case_outcome.value_counts().plot(kind="bar")
fig.update_xaxes(title="Case outcome")
fig.update_yaxes(title="No. of cases")

project["data/distribution"].upload(fig)

# ## Data processing


def clean_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cleans a dataframe `df` string column `col` by applying the following transformations:
    * Convert string to lower-case
    * Remove punctuation
    * Remove numbers
    * Remove single-letter words
    * Remove stopwords
    * Remove multiple and leading/trailing whitespaces

    Args:
        df: Dataframe containing sgtring columns `col` to be cleaned
        col: String column to be cleaned

    Returns:
        A copy of the dataframe `df` with the column `col` cleaned
    """

    tqdm.pandas()
    stop = set(stopwords.words("english"))
    pat = f'\b(?:{"|".join(stop)})\b'

    _df = df.copy()
    _df[col] = (
        df[col]
        .progress_apply(str.lower)  # Converting to lowercase
        .progress_apply(lambda x: re.sub(r"[^\w\s]", " ", x))  # Removing punctuation
        .progress_apply(
            lambda x: " ".join(x for x in x.split() if not any(c.isdigit() for c in x))
        )  # Removing numbers
        .progress_apply(lambda x: re.sub(r"\b\w\b", "", x))  # Removing single-letter words
        .str.replace(pat, "", regex=True)  # Removing stopwords
        .progress_apply(lambda x: re.sub(r" +", " ", x))  # Removing multiple-whitespaces
        .str.strip()  # Removing leading and whitepaces
    )

    return _df


df_fasttext_raw = df_raw[["case_outcome", "case_text"]]
df_fasttext_raw["label"] = "__label__" + df_fasttext_raw.case_outcome.str.replace(" ", "_")
df_fasttext_raw = df_fasttext_raw[["label", "case_text"]]

DATASET_PATH_LOCAL = path.cwd().parent.parent.joinpath("data")

if not os.path.exists(DATASET_PATH_LOCAL):
    os.makedirs(DATASET_PATH_LOCAL)

DATASET_PATH_LOCAL_FASTTEXT = DATASET_PATH_LOCAL.joinpath("fasttext")

if not os.path.exists(DATASET_PATH_LOCAL_FASTTEXT):
    os.makedirs(DATASET_PATH_LOCAL_FASTTEXT)

TO_CSV_KWARGS = {
    "sep": " ",
    "header": False,
    "index": False,
    "quoting": csv.QUOTE_NONE,
    "quotechar": "",
    "escapechar": " ",
}

df_fasttext_raw.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("raw.txt"), **TO_CSV_KWARGS)

df_processed = clean_text(df_fasttext_raw, "case_text")
df_processed.drop_duplicates(subset="case_text", inplace=True)

df_processed.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("processed.txt"), **TO_CSV_KWARGS)

X = df_processed["case_text"]
y = df_processed["label"]

X_train, X_, y_train, y_ = train_test_split(X, y, stratify=y, train_size=0.7)
X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, stratify=y_, train_size=0.5)

print(f"Training size: {X_train.shape}")
print(f"Validation size: {X_valid.shape}")
print(f"Test size: {X_test.shape}")

df_train = pd.DataFrame(data=[y_train, X_train]).T
df_valid = pd.DataFrame(data=[y_valid, X_valid]).T
df_test = pd.DataFrame(data=[y_test, X_test]).T

df_train.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("train.txt"), **TO_CSV_KWARGS)
df_valid.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("valid.txt"), **TO_CSV_KWARGS)
df_test.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("test.txt"), **TO_CSV_KWARGS)

# ## (Neptune) Initialize an optuna study-level run
# A run is a namespace inside a project where you log metadata. Typically, you create a run every time you execute a script that does model training, re-training, or inference.
# [Read the docs](https://docs.neptune.ai/you-should-know/core-concepts#run)

run = neptune.init_run(
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
    name="Fasttext text classification",
    description="Optuna tuned fasttext text classification",
    tags=["fasttext", "optuna", "study-level", "notebook", "sagemaker"],
)

# ### (Neptune) Track run-specific files
# These are the files which are created during the run, and should be tracked within the run and not the project.
# [Read the docs](https://docs.neptune.ai/how-to-guides/data-versioning/compare-datasets#step-2-add-tracking-of-the-dataset-version)

run["data/files"].track_files(os.path.relpath(DATASET_PATH_LOCAL_FASTTEXT))

csv_buffer = StringIO()

df_fasttext_raw.sample(100).to_csv(csv_buffer, index=False)
run["data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))

# ### (Neptune) Log metadata to run
# You can log nested dictionaries to create custom nested namespaces.
# [Read the docs](https://docs.neptune.ai/you-should-know/logging-metadata)

metadata = {
    "train_size": len(df_train),
    "test_size": len(df_test),
}

run["data/metadata"] = metadata

# ### (Neptune) Log sweep and trial parameters
# Neptune's Optuna integration lets you log metadata from both the study-level and trial-level runs.
# [Read the docs](https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/optuna)

import uuid

sweep_id = uuid.uuid1()
print(f"Optuna sweep-id: {sweep_id}")

run["study/sweep_id"] = sweep_id


def objective_with_logging(trial: optuna.trial.Trial) -> float64:
    """Optuna objective function with inbuilt Neptune tracking

    Args:
        trial (_type_): _description_

    Returns:
        _type_: _description_
    """
    params = {
        "lr": trial.suggest_float("lr", 0.1, 1, step=0.1),
        "dim": trial.suggest_int("dim", 10, 1000, log=True),
        "ws": trial.suggest_int("ws", 1, 10),
        "epoch": trial.suggest_int("epoch", 10, 100),
        "minCount": trial.suggest_int("minCount", 1, 10),
        "wordNgrams": trial.suggest_int("wordNgrams", 1, 3),
        "loss": trial.suggest_categorical("loss", ["hs", "softmax", "ova"]),
        "bucket": trial.suggest_int("bucket", 1000000, 3000000, log=True),
        "lrUpdateRate": trial.suggest_int("lrUpdateRate", 1, 10),
        "t": trial.suggest_float("t", 0.00001, 0.1, log=True),
    }

    # (Neptune) create a trial-level Run
    run_trial_level = neptune.init_run(
        project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
        name="Fasttext text classification",
        description="Optuna tuned fasttext text classification",
        tags=["fasttext", "optuna", "trial-level", "notebook", "sagemaker"],
    )

    # (Neptune) log sweep id to trial-level Run
    run_trial_level["sweep_id"] = sweep_id

    # train model with chosen hyperparameters
    clf = fasttext.train_supervised(
        input=str(DATASET_PATH_LOCAL_FASTTEXT.joinpath("train.txt")),
        verbose=0,
        **params,
    )

    # (Neptune) log parameters of a trial-level Run
    properties = {k: v for k, v in vars(clf).items() if k not in ["_words", "f"]}
    run_trial_level["model/properties"] = properties

    # make predictions and calculate metrics
    preds = [clf.predict(text)[0][0] for text in X_valid.values]

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_valid,
        preds,
        average="weighted",
        zero_division=0,
    )

    # (Neptune) log metrics for current hyperparameters
    run_trial_level["validation/metrics/precision"] = precision
    run_trial_level["validation/metrics/recall"] = recall
    run_trial_level["validation/metrics/f1_score"] = f1_score

    # (Neptune) stop trial-level Run
    run_trial_level.stop()

    return f1_score


# #### (Neptune) Create the Neptune callback for Optuna and pass it to the Optuna study object
import neptune.new.integrations.optuna as optuna_utils

neptune_callback = optuna_utils.NeptuneCallback(run)

study = optuna.create_study(direction="maximize")
study.optimize(
    objective_with_logging,
    n_trials=N_TRIALS,
    callbacks=[neptune_callback],
    n_jobs=N_JOBS,
)

# ### (Neptune) Register a model
# With Neptune's model registry, you can store your ML models in a central location and collaboratively manage their lifecycle.
# [Read the docs](https://docs.neptune.ai/how-to-guides/model-registry)

model = neptune.init_model(
    with_id="TXTCLF-FTXT",  # Reinitializing an existing model
    # name="fasttext", # Required only for new models
    # key="FTXT", # Required only for new models
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
)

# #### (Neptune) Create a new model version
# For each model, you can create different versions as you refine the model.
# [Read the docs](https://docs.neptune.ai/how-to-guides/model-registry/creating-model-versions)

model_version = neptune.init_model_version(
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
    model=model.get_structure()["sys"]["id"].fetch(),
)

# #### (Neptune) Associate model version to run and vice-versa
# This is to help find the model created by the run in the runs table, and the run which created the model in the models table.

run_dict = {
    "id": run.get_structure()["sys"]["id"].fetch(),
    "name": run.get_structure()["sys"]["name"].fetch(),
    "url": run.get_url(),
}

model_version["run"] = run_dict

model_version_dict = {
    "id": model_version.get_structure()["sys"]["id"].fetch(),
    "url": model_version.get_url(),
}

run["model"] = model_version_dict

# #### Train model based on Hyperparameters chosen by Optuna

clf = fasttext.train_supervised(
    input=str(DATASET_PATH_LOCAL_FASTTEXT.joinpath("train.txt")),
    verbose=5,
    **study.best_params,
)

# #### (Neptune) Upload serialized model to model registry
# Similar to artifact tracking in a project/run, you can also track a pointer to the model in the model registry, or upload the entire serialized model object as well.
# [Read the docs](https://docs.neptune.ai/how-to-guides/model-registry/creating-model-versions)

MODEL_PATH = path.cwd().parent.parent.joinpath("models")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

from datetime import datetime

MODEL_NAME = str(MODEL_PATH.joinpath(f"fasttext_{datetime.now().strftime('%Y%m%d%H%M%S')}.bin"))
clf.save_model(MODEL_NAME)

if os.path.getsize(MODEL_NAME) < 1024 * 1024 * UPLOAD_SIZE_THRESHOLD:  # 100 MB
    print("Uploading serialized model")
    model_version["serialized_model"].upload(MODEL_NAME)
else:
    print(
        f"Model is larger than UPLOAD_SIZE_THRESHOLD ({UPLOAD_SIZE_THRESHOLD} MB). Tracking pointer to model file"
    )
    model_version["serialized_model"].track_files(os.path.relpath(MODEL_NAME))

# #### (Neptune) Log model properties to model_version
# Neptune dynamically creates nested namespaces based on the dictionary structure.

properties = {k: v for k, v in vars(clf).items() if k not in ["_words", "f"]}

model_version["properties"] = properties

# ### Make predictions

preds = [clf.predict(text)[0][0] for text in X_test.values]

df_test["prediction"] = preds

# ### (Neptune) Log parameters, metrics and debugging information to run and model version

precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test,
    preds,
    average="weighted",
    zero_division=0,
)

print(f"Precision: {precision}\nRecall: {recall}\nF1-score: {f1_score}")

run["test/metrics/precision"] = model_version["metrics/precision"] = precision
run["test/metrics/recall"] = model_version["metrics/recall"] = recall
run["test/metrics/f1_score"] = model_version["metrics/f1_score"] = f1_score

print(classification_report(y_test, preds, zero_division=0))

# (Neptune) Log each metric in its separate nested namespace
run["test/metrics/classification_report"] = classification_report(
    y_test, preds, output_dict=True, zero_division=0
)

# (Neptune) Log classification report as an HTML dataframe
df_clf_rpt = pd.DataFrame(classification_report(y_test, preds, output_dict=True, zero_division=0)).T
run["test/metrics/classification_report/report"].upload(File.as_html(df_clf_rpt))

fig = ConfusionMatrixDisplay.from_predictions(
    y_test, preds, xticks_rotation="vertical", colorbar=False
)
run["test/debug/plots/confusion_matrix"].upload(fig.figure_)

labels = [s.replace("__label__", "") for s in df_test.label.value_counts().index]
fig = go.Figure(
    data=[
        go.Bar(name="Actual", x=labels, y=df_test.label.value_counts()),
        go.Bar(name="Prediction", x=labels, y=df_test.prediction.value_counts()),
    ]
)
fig.update_layout(title="Actual vs Prediction", barmode="group")

run["test/debug/plots/prediction_distribution"].upload(fig)

# ### (Neptune) Log misclassified results
# CSV files logged to Neptune will be rendered as an interactive table.
# [Read the docs](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#csv-files)

df_debug = df_test[df_test.label != df_test.prediction]

csv_buffer = StringIO()

df_debug.to_csv(csv_buffer, index=False)
run["test/debug/misclassifications"].upload(File.from_stream(csv_buffer, extension="csv"))

# ## (Neptune) Explore the [project](https://app.neptune.ai/showcase/project-text-classification) in the Neptune app
