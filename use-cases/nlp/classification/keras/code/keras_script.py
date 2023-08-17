#!/usr/bin/env python
# coding: utf-8

#########################################################
# Text classification using Keras with Neptune tracking #
#########################################################
# Script inspired from https://keras.io/examples/nlp/text_classification_from_scratch/

#########
# Setup #
#########

import os

# Import requirements
import random

import neptune
import tensorflow as tf
from neptune.exceptions import NeptuneModelKeyAlreadyExistsError
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.utils import stringify_unsupported
from tensorflow.keras.layers import TextVectorization


# Create utility functions
def extract_files(source: str, destination: str) -> None:
    """Extracts files from the source archive to the destination path

    Args:
        source (str): Archive file path
        destination (str): Extract destination path
    """

    import tarfile

    print("Extracting data...")
    with tarfile.open(source) as f:
        f.extractall(destination)


def prep_data(imdb_folder: str, dest_path: str) -> None:
    """Removes unnecessary folders/files and renames source folder

    Args:
        imdb_folder (str): Path of the aclImdb folder
        dest_name (str): Destination folder to which the aclImdb folder has to be renamed to
    """
    import os
    import shutil

    shutil.rmtree(f"{imdb_folder}/train/unsup")
    os.remove(f"{imdb_folder.rsplit('/', maxsplit=1)[0]}/aclImdb_v1.tar.gz")

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    os.rename(imdb_folder, dest_path)
    print(f"{imdb_folder} renamed to {dest_path}")


def custom_standardization(input_data):
    import re
    import string

    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]", "")


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def build_model(model_params: dict, data_params: dict):
    """Accepts model and data parameters to build and compile a keras model

    Args:
        model_params (dict): Model parameters
        data_params (dict): Data parameters

    Returns:
        A compiled keras model
    """

    import tensorflow as tf
    from tensorflow.keras import layers

    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(data_params["max_features"], data_params["embedding_dim"])(inputs)
    x = layers.Dropout(model_params["dropout"])(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(
        data_params["embedding_dim"],
        model_params["kernel_size"],
        padding="valid",
        activation=model_params["activation"],
        strides=model_params["strides"],
    )(x)
    x = layers.Conv1D(
        data_params["embedding_dim"],
        model_params["kernel_size"],
        padding="valid",
        activation=model_params["activation"],
        strides=model_params["strides"],
    )(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(data_params["embedding_dim"], activation=model_params["activation"])(x)
    x = layers.Dropout(model_params["dropout"])(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    keras_model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    keras_model.compile(
        loss=model_params["loss"],
        optimizer=model_params["optimizer"],
        metrics=model_params["metrics"],
    )

    return keras_model


#####################################################
# (Neptune) Import Neptune and initialize a project #
#####################################################
os.environ["NEPTUNE_PROJECT"] = "common/project-text-classification"

project = neptune.init_project()

####################
# Data preparation #
####################
project["keras/data/files"].track_files(
    "s3://neptune-examples/data/text-classification/aclImdb_v1.tar.gz"
)
project.wait()


# (Neptune) Download files from S3 using Neptune
print("Downloading data...")
project["keras/data/files"].download("..")


# Prepare data
extract_files(source="../aclImdb_v1.tar.gz", destination="..")
prep_data(
    imdb_folder="../aclImdb", dest_path="../data"
)  # If you get a permission error here, you can manually rename the `aclImdb` folder to `data`


# (Neptune) Upload dataset sample to Neptune project

base_namespace = "keras/data/sample/"

project[base_namespace]["train/pos"].upload(
    f"../data/train/pos/{random.choice(os.listdir('../data/train/pos'))}"
)
project[base_namespace]["train/neg"].upload(
    f"../data/train/neg/{random.choice(os.listdir('../data/train/neg'))}"
)
project[base_namespace]["test/pos"].upload(
    f"../data/test/pos/{random.choice(os.listdir('../data/test/pos'))}"
)
project[base_namespace]["test/neg"].upload(
    f"../data/test/neg/{random.choice(os.listdir('../data/test/neg'))}"
)

##############################
# (Neptune) Initialize a run #
##############################

run = neptune.init_run(
    name="Keras text classification",
    tags=["keras", "script"],
    dependencies="requirements.txt",
)


# (Neptune) Log data metadata to run
data_params = {
    "batch_size": 64,
    "validation_split": 0.3,
    "max_features": 2000,
    "embedding_dim": 64,
    "sequence_length": 500,
    "seed": 42,
}

run["data/params"] = data_params


# (Neptune) Track dataset at the run-level
run["data/files"] = project["keras/data/files"].fetch()


# Generate training, validation, and test datasets
raw_train_ds, raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../data/train",
    batch_size=data_params["batch_size"],
    validation_split=data_params["validation_split"],
    subset="both",
    seed=data_params["seed"],
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../data/test", batch_size=data_params["batch_size"]
)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")


# Clean data
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=data_params["max_features"],
    output_mode="int",
    output_sequence_length=data_params["sequence_length"],
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


# Vectorize data

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

#############
# Modelling #
#############

# (Neptune) Register a model and create a new model version

project_key = project["sys/id"].fetch()
model_key = "KER"

try:
    model = neptune.init_model(name="keras", key=model_key)
    model.stop()
except NeptuneModelKeyAlreadyExistsError:
    # If it already exists, we don't have to do anything.
    pass

model_version = neptune.init_model_version(model=f"{project_key}-{model_key}", name="keras")


# Build a model
model_params = {
    "dropout": 0.5,
    "strides": 5,
    "activation": "relu",
    "kernel_size": 3,
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],
}

model_version["params/model"] = run["training/model/params"] = stringify_unsupported(model_params)
model_version["params/data"] = data_params

keras_model = build_model(model_params, data_params)


###### Train the model ######

# (Neptune) Initialize the Neptune callback

neptune_callback = NeptuneCallback(run=run, log_model_diagram=False, log_on_batch=True)

training_params = {
    "epochs": 2,
}

# Fit the model using the train and test datasets.
keras_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_params["epochs"],
    callbacks=neptune_callback,
)

# Evaluate the model

# We save the accuracy of the  model to be able to evaluate it against the champion model in production later in the code
_, curr_model_acc = keras_model.evaluate(test_ds, callbacks=neptune_callback)


# (Neptune) Associate run with model and vice-versa #

run_meta = {
    "id": run["sys/id"].fetch(),
    "name": run["sys/name"].fetch(),
    "url": run.get_url(),
}

print(run_meta)

model_version["run"] = run_meta

model_version_meta = {
    "id": model_version["sys/id"].fetch(),
    "name": model_version["sys/name"].fetch(),
    "url": model_version.get_url(),
}

print(model_version_meta)

run["training/model/meta"] = model_version_meta


# (Neptune) Upload serialized model and model weights to Neptune
model_version["serialized_model"] = keras_model.to_json()

keras_model.save_weights("model_weights.h5")
model_version["model_weights"].upload("model_weights.h5")


# (Neptune) Update model stage
model_version.change_stage("staging")

model_version.wait()

##############################################
# (Neptune) Promote best model to production #
##############################################

# (Neptune) Fetch current champion model
with neptune.init_model(with_id=f"{project_key}-KER") as model:
    model_versions_df = model.fetch_model_versions_table().to_pandas()

production_models = model_versions_df[model_versions_df["sys/stage"] == "production"]["sys/id"]
# assert (
#     len(production_models) == 1
# ), f"Multiple model versions found in production: {production_models.values}"

prod_model_id = production_models.values[0]
print(f"Current champion model: {prod_model_id}")

npt_prod_model = neptune.init_model_version(with_id=prod_model_id)
npt_prod_model_params = npt_prod_model["params/model"].fetch()
prod_model = tf.keras.models.model_from_json(npt_prod_model["serialized_model"].fetch())

npt_prod_model["model_weights"].download()
prod_model.load_weights("model_weights.h5")


#####  (Neptune) Evaluate current model on lastest test data #####

# (Neptune) Fetch data parameters from the current champion model to preserve data preprocessing
prod_data_params = npt_prod_model["params/data"].fetch()

print(prod_data_params)


# Preparing test data according to fetched data parameters
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../data/test", batch_size=prod_data_params["batch_size"]
)

print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=prod_data_params["max_features"],
    output_mode="int",
    output_sequence_length=prod_data_params["sequence_length"],
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

test_ds = raw_test_ds.map(vectorize_text)
test_ds = test_ds.cache().prefetch(buffer_size=10)

# Evaluate champion model using the model's original loss and optimizer, but the current metric
prod_model.compile(
    loss=npt_prod_model_params["loss"],
    optimizer=npt_prod_model_params["optimizer"],
    metrics=model_params["metrics"],
)

_, prod_model_acc = prod_model.evaluate(test_ds)


# (Neptune) If challenger model outperforms production model, promote it to production and mark it's run as the new `prod` run

print(f"Champion model accuracy: {prod_model_acc}\nChallenger model accuracy: {curr_model_acc}")

if curr_model_acc > prod_model_acc:
    print("Promoting challenger to champion")
    npt_prod_model.change_stage("archived")
    model_version.change_stage("production")
else:
    print("Archiving challenger model")
    model_version.change_stage("archived")


###########################
# (Neptune) Stop tracking #
###########################
npt_prod_model.stop()
model_version.stop()
run.stop()
project.stop()
