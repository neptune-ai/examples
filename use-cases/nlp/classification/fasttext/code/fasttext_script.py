#!/usr/bin/env python
# coding: utf-8

############################################################
# Text classification using fastText with Neptune tracking #
############################################################

##### Setup #####

# Import dependencies

import csv
import os
import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import fasttext
import neptune
import nltk
import pandas as pd
import plotly.graph_objects as go
from neptune.types import File
from neptune.utils import stringify_unsupported
from nltk.corpus import stopwords
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
pd.options.plotting.backend = "plotly"
path = Path()


# Set parameters
UPLOAD_SIZE_THRESHOLD = 50  # in MB

##### Load input files #####

DATASET_PATH_S3 = "s3://neptune-examples/data/text-classification/legal_text_classification.csv"

df_raw = pd.read_csv(DATASET_PATH_S3)
df_raw.dropna(subset=["case_text"], inplace=True)
df_raw.drop_duplicates(subset="case_text", inplace=True)


##########################################
# (Neptune) Initialize a neptune project #
##########################################

WORKSPACE_NAME = "common"
PROJECT_NAME = "project-text-classification"

os.environ["NEPTUNE_PROJECT"] = f"{WORKSPACE_NAME}/{PROJECT_NAME}"

project = neptune.init_project()


##### (Neptune) Log project level metadata #####
project["fasttext/data/files"].track_files(DATASET_PATH_S3)


##### (Neptune) Log dataset sample #####
csv_buffer = StringIO()
df_raw.sample(100).to_csv(csv_buffer, index=False)
project["fasttext/data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))


##### (Neptune) Log metadata plots #####
fig = df_raw.case_outcome.value_counts().plot(kind="bar")
fig.update_xaxes(title="Case outcome")
fig.update_yaxes(title="No. of cases")
project["fasttext/data/distribution"].upload(fig)

###################
# Data processing #
###################


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

    stop = set(stopwords.words("english"))
    pat = f'\b(?:{"|".join(stop)})\b'

    _df = df.copy()
    _df[col] = (
        df[col]
        .apply(str.lower)  # Converting to lowercase
        .apply(lambda x: re.sub(r"[^\w\s]", " ", x))  # Removing punctuation
        .apply(
            lambda x: " ".join(x for x in x.split() if not any(c.isdigit() for c in x))
        )  # Removing numbers
        .apply(lambda x: re.sub(r"\b\w\b", "", x))  # Removing single-letter words
        .str.replace(pat, "", regex=True)  # Removing stopwords
        .apply(lambda x: re.sub(r" +", " ", x))  # Removing multiple-whitespaces
        .str.strip()  # Removing leading and whitepaces
    )

    return _df


df_fasttext_raw = df_raw[["case_outcome", "case_text"]]
df_fasttext_raw["label"] = "__label__" + df_fasttext_raw.case_outcome.str.replace(" ", "_")
df_fasttext_raw = df_fasttext_raw[["label", "case_text"]]

DATASET_PATH_LOCAL = path.cwd().parent.joinpath("data")

if not os.path.exists(DATASET_PATH_LOCAL):
    os.makedirs(DATASET_PATH_LOCAL)

if not os.path.exists(DATASET_PATH_LOCAL):
    os.makedirs(DATASET_PATH_LOCAL)

TO_CSV_KWARGS = {
    "sep": " ",
    "header": False,
    "index": False,
    "quoting": csv.QUOTE_NONE,
    "quotechar": "",
    "escapechar": " ",
}

df_fasttext_raw.to_csv(DATASET_PATH_LOCAL.joinpath("raw.txt"), **TO_CSV_KWARGS)

df_processed = clean_text(df_fasttext_raw, "case_text")
df_processed.drop_duplicates(subset="case_text", inplace=True)
df_processed.to_csv(DATASET_PATH_LOCAL.joinpath("processed.txt"), **TO_CSV_KWARGS)

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

df_train.to_csv(DATASET_PATH_LOCAL.joinpath("train.txt"), **TO_CSV_KWARGS)
df_valid.to_csv(DATASET_PATH_LOCAL.joinpath("valid.txt"), **TO_CSV_KWARGS)
df_test.to_csv(DATASET_PATH_LOCAL.joinpath("test.txt"), **TO_CSV_KWARGS)

##############################
# (Neptune) Initialize a run #
##############################

run = neptune.init_run(
    name="Fasttext",
    description="Fasttext text classification",
    tags=["fasttext", "autotuned", "script"],
)


##### (Neptune) Track run-specific files #####
run["data/files"].track_files(os.path.relpath(DATASET_PATH_LOCAL))

csv_buffer = StringIO()
df_fasttext_raw.sample(100).to_csv(csv_buffer, index=False)
run["data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))

##### (Neptune) Log metadata to run #####
metadata = {
    "train_size": len(df_train),
    "test_size": len(df_test),
}

run["data/metadata"] = metadata


##########################
# Train a fasttext model #
##########################

##### Setup autotune #####

AUTOTUNE_PARAMS = {
    "autotuneDuration": 60,
    "autotuneModelSize": "100M",
}

run["model/params"] = AUTOTUNE_PARAMS


##### Train model #####

clf = fasttext.train_supervised(
    input=str(DATASET_PATH_LOCAL.joinpath("train.txt")),
    autotuneValidationFile=str(DATASET_PATH_LOCAL.joinpath("valid.txt")),
    **AUTOTUNE_PARAMS,
)


##### (Neptune) Log model properties to the run #####

properties = {k: v for k, v in vars(clf).items() if k not in ["_words", "f"]}
run["model/properties"] = stringify_unsupported(properties)

##############################################
# Make predictions and calculate the metrics #
##############################################

preds = [clf.predict(text)[0][0] for text in X_valid.values]

precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_valid,
    preds,
    average="weighted",
    zero_division=0,
)

##### (Neptune) Log metrics to the run #####

run["validation/metrics/precision"] = precision
run["validation/metrics/recall"] = recall
run["validation/metrics/f1_score"] = f1_score


#############################################################
# (Neptune) Register a model and create a new model version #
#############################################################

with neptune.init_model(
    with_id="TXTCLF-FTXT",  # Reinitializing an existing model. Not required for new models
    # name="fasttext", # Required only for new models
    # key="FTXT", # Required only for new models
) as model:
    model_version = neptune.init_model_version(model=model["sys/id"].fetch())
    model_version.change_stage("staging")


##### (Neptune) Associate model version to run and vice-versa #####

run_dict = {
    "id": run["sys/id"].fetch(),
    "name": run["sys/name"].fetch(),
    "url": run.get_url(),
}
model_version["run"] = run_dict

model_version_dict = {
    "id": model_version["sys/id"].fetch(),
    "url": model_version.get_url(),
}
run["model"] = model_version_dict

##### (Neptune) Upload serialized model to model registry #####

MODEL_PATH = path.cwd().parent.joinpath("models")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

MODEL_NAME = str(MODEL_PATH.joinpath(f"fasttext_{datetime.now().strftime('%Y%m%d%H%M%S')}.bin"))
clf.save_model(MODEL_NAME)

if os.path.getsize(MODEL_NAME) < 1024 * 1024 * UPLOAD_SIZE_THRESHOLD:
    print("Uploading serialized model")
    model_version["serialized_model"].upload(MODEL_NAME)
else:
    print(
        f"Model is larger than UPLOAD_SIZE_THRESHOLD ({UPLOAD_SIZE_THRESHOLD} MB). Tracking pointer to model file"
    )
    model_version["serialized_model"].track_files(os.path.relpath(MODEL_NAME))

##### (Neptune) Copy model properties from run to model version #####
model_version["properties"] = run["model/properties"].fetch()

##### Make predictions #####
preds = [clf.predict(text)[0][0] for text in X_test.values]
df_test["prediction"] = preds

########################################################################################
# (Neptune) Log parameters, metrics and debugging information to run and model version #
########################################################################################

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

##### (Neptune) Log each metric in its separate nested namespace #####
run["test/metrics/classification_report"] = classification_report(
    y_test, preds, output_dict=True, zero_division=0
)

##### (Neptune) Log classification report as an HTML dataframe #####
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


##### (Neptune) Log misclassified results #####
df_debug = df_test[df_test.label != df_test.prediction]

csv_buffer = StringIO()
df_debug.to_csv(csv_buffer, index=False)
run["test/debug/misclassifications"].upload(File.from_stream(csv_buffer, extension="csv"))

####################################################
# (Neptune) Compare challenger model with champion #
####################################################

##### (Neptune) Fetch current production model #####

with neptune.init_model(with_id=f"{project['sys/id'].fetch()}-FTXT") as model:
    model_versions_df = model.fetch_model_versions_table().to_pandas()

production_models = model_versions_df[model_versions_df["sys/stage"] == "production"]["sys/id"]
prod_model_id = production_models.values[0]
print(f"Current model in production: {prod_model_id}")

npt_prod_model = neptune.init_model_version(with_id=prod_model_id)
npt_prod_model["serialized_model"].download()
prod_model = fasttext.load_model("serialized_model.bin")

##### (Neptune) Evaluate current model on lastest test data #####

preds = [prod_model.predict(text)[0][0] for text in X_test.values]

_, _, prod_f1_score, _ = precision_recall_fscore_support(
    y_test,
    preds,
    average="weighted",
    zero_division=0,
)

##### (Neptune) If challenger model outperforms production model, promote it to production #####
print(f"Production model score: {prod_f1_score}\nChallenger model score: {f1_score}")

if f1_score > prod_f1_score:
    print("Promoting challenger to production")
    npt_prod_model.change_stage("archived")
    model_version.change_stage("production")
else:
    print("Archiving challenger model")
    model_version.change_stage("archived")

npt_prod_model.stop()


##################################
# (Neptune) Stop Neptune objects #
##################################
model_version.stop()
run.stop()
project.stop()
