import os
from typing import List

import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


def save_dataset(X: List, y: List, filename: str):
    path_to_file = f"./{filename}"
    np.savez(path_to_file, x=np.array(X), y=np.array(y))
    print(f"File saved on {path_to_file}")


def load_dataset(filename: str):
    path_to_file = f"./{filename}"
    if os.path.exists(path_to_file):
        return np.load(path_to_file)
    else:
        print(f"File doesn't exist on {path_to_file}")


class Preprocessing:
    def __init__(self, dataset):

        # introspect the images arrays to find the shapes (for plotting)
        self.n_samples, _, _ = dataset.images.shape

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        self.X = dataset.data
        self.n_features = self.X.shape[1]

        # the label to predict is the id of the person
        self.y = dataset.target
        self.target_names = dataset.target_names
        self.n_classes = self.target_names.shape[0]

    def scale_features(self, data_filename: str = "data", scaler_filename: str = "data_scaler"):

        # Feature scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(self.X)

        # Save scaled features and scaler to disk
        save_dataset(X, self.y, data_filename)
        dump(scaler, f"./{scaler_filename}.joblib")

    def describe(self):
        print("====================================")
        print("Total dataset size")
        print("n_samples: %d" % self.n_samples)
        print("n_features: %d" % self.n_features)
        print("n_classes: %d" % self.n_classes)
