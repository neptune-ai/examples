import glob
import os
import uuid

import neptune
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from arize.api import Client
from arize.utils.types import Environments, ModelTypes, Schema
from keras.layers import Dense, Dropout
from keras.models import Sequential
from neptune.integrations.tensorflow_keras import NeptuneCallback
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

model_id = "neptune_cancer_prediction_model"
model_version = "v1"
model_type = ModelTypes.BINARY_CLASSIFICATION


def process_data(X, y):
    X = np.array(X).reshape((len(X), 30))
    y = np.array(y)
    return X, y


# Load data and split data
data = datasets.load_breast_cancer()

X, y = datasets.load_breast_cancer(return_X_y=True)
X, y = X.astype(np.float32), y

X, y = pd.DataFrame(X, columns=data["feature_names"]), pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42)


# Define and compile model
model = Sequential()
model.add(Dense(10, activation="sigmoid", input_shape=((30,))))
model.add(Dropout(0.25))
model.add(Dense(20, activation="sigmoid"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="sigmoid"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.mean_squared_logarithmic_error,
)

# Fit model and log callbacks
params = {
    "batch_size": 30,
    "epochs": 50,
    "verbose": 0,
}

# (Neptune) Initialize a run
run = neptune.init_run(project="common/showroom", api_token=neptune.ANONYMOUS_API_TOKEN)

callbacked = model.fit(
    X_train,
    y_train,
    batch_size=params["batch_size"],
    epochs=params["epochs"],
    verbose=params["verbose"],
    validation_data=(X_test, y_test),
    # (Neptune) log to Neptune using a Neptune callback
    callbacks=[NeptuneCallback(run=run)],
)

# Storing model version 1
directory_name = f"keras_model_{model_version}"
model.save(directory_name)

run[f"{directory_name}/saved_model.pb"].upload(f"{directory_name}/saved_model.pb")

for name in glob.glob(f"{directory_name}/variables/*"):
    run[name].upload(name)

# (Neptune) Log "model_id", for better reference
run["model_id"] = model_id

# (Arize) Initialize Arize client
arize = Client(space_key=os.environ["ARIZE_SPACE_KEY"], api_key=os.environ["API_KEY"])

# Use the model to generate predictions
y_train_pred = model.predict(X_train).T[0]
y_val_pred = model.predict(X_val).T[0]
y_test_pred = model.predict(X_test).T[0]

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
