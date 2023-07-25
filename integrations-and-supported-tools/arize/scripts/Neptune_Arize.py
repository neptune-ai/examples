import os

import neptune
import neptune.integrations.sklearn as npt_utils
import numpy as np
import pandas as pd
from arize.api import Client
from arize.utils.types import Environments, ModelTypes
from neptune.utils import stringify_unsupported
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model_id = "neptune_cancer_prediction_model"
model_version = "v1"
model_type = ModelTypes.BINARY_CLASSIFICATION


def process_data(X, y):
    X = np.array(X).reshape((len(X), 30))
    y = np.array(y)
    return X, y


# Load and split data
data = datasets.load_breast_cancer()

X, y = datasets.load_breast_cancer(return_X_y=True)
X, y = X.astype(np.float32), y

X, y = pd.DataFrame(X, columns=data["feature_names"]), pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42)

# Define model
model = LogisticRegression(random_state=42, max_iter=1000)

# Fit model and log callbacks
params = {
    "batch_size": 30,
    "epochs": 50,
    "verbose": 0,
}

# (Neptune) Initialize a run
run = neptune.init_run(project="common/showroom", api_token=neptune.ANONYMOUS_API_TOKEN)

# Model training
model.fit(X_train, y_train)

# (Neptune) Log model performance
run["regression_summary"] = npt_utils.create_classifier_summary(
    model, X_train, X_test, y_train, y_test
)

# (Neptune) Log model parameters
run["estimator/params"] = stringify_unsupported(npt_utils.get_estimator_params(model))

# (Neptune) Save model
run["estimator/pickled-model"] = npt_utils.get_pickled_model(model)

# (Neptune) Log "model_id", for better reference
run["model_id"] = model_id

# (Arize) Initialize Arize client
arize = Client(space_key=os.environ["ARIZE_SPACE_KEY"], api_key=os.environ["ARIZE_API_KEY"])

# Generate model predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# (Arize) Logging training
train_prediction_labels = pd.Series(y_train_pred, name="predicted_labels")
train_actual_labels = pd.Series(y_train, name="actual_labels")
train_feature_df = pd.DataFrame(X_train, columns=data["feature_names"]).to_dict("list")

train_responses = arize.log(
    model_id=model_id,
    model_version=model_version,
    model_type=model_type,
    prediction_label=train_prediction_labels,
    actual_label=train_actual_labels,
    environment=Environments.TRAINING,
    features=train_feature_df,
)

# (Arize) Logging validation
val_prediction_labels = pd.Series(y_val_pred)
val_actual_labels = pd.Series(y_val)
val_features_df = pd.DataFrame(X_val, columns=data["feature_names"]).to_dict("list")

val_responses = arize.log(
    model_id=model_id,
    model_version=model_version,
    model_type=model_type,
    batch_id="batch0",
    prediction_label=val_prediction_labels,
    actual_label=val_actual_labels,
    environment=Environments.VALIDATION,
    features=val_features_df,
)
