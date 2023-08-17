import neptune
from sklearn.datasets import fetch_lfw_people

from utils import *

# Download dataset
dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# (Neptune) Create a new run
run = neptune.init_run(
    monitoring_namespace="monitoring/preprocessing",
)

# Get dataset details
dataset_config = {
    "target_names": str(dataset.target_names.tolist()),
    "n_classes": dataset.target_names.shape[0],
    "n_samples": dataset.images.shape[0],
    "height": dataset.images.shape[1],
    "width": dataset.images.shape[2],
}

# (Neptune) Set up "preprocessing" namespace inside the run.
# This will be the base namespace where all the preprocessing metadata is logged.
preprocessing_handler = run["preprocessing"]

# (Neptune) Log dataset details
preprocessing_handler["dataset/config"] = dataset_config

# Preprocess dataset
dataset_transform = Preprocessing(
    dataset,
    dataset_config["n_samples"],
    dataset_config["target_names"],
    dataset_config["n_classes"],
    (dataset_config["height"], dataset_config["width"]),
)
path_to_scaler = dataset_transform.scale_data()
path_to_features = dataset_transform.create_and_save_features(data_filename="features")
dataset_transform.describe()

# (Neptune) Log scaler and features files
preprocessing_handler["dataset/scaler"].upload(path_to_scaler)
preprocessing_handler["dataset/features"].upload(path_to_features)
