import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def save_model(model, filename) -> str:
    path_to_file = f"./{filename}.joblib"
    dump(model, path_to_file)
    return path_to_file


def load_model(path_to_file):
    return load(path_to_file)


def save_dataset(
    X_train: List,
    y_train: List,
    X_test: List,
    y_test: List,
    X_train_pca: List,
    X_test_pca: List,
    eigen_faces: List,
    h: int,
    w: int,
    target_names: List,
    n_classes: int,
    filename: str,
) -> str:
    path_to_file = f"./{filename}"
    np.savez(
        path_to_file,
        x_train=np.array(X_train),
        y_train=np.array(y_train),
        x_test=np.array(X_test),
        y_test=np.array(y_test),
        x_train_pca=np.array(X_train_pca),
        x_test_pca=np.array(X_test_pca),
        eigen_faces=np.array(eigen_faces),
        h=h,
        w=w,
        target_names=target_names,
        n_classes=n_classes,
    )
    print(f"File saved on {path_to_file}")

    return f"{path_to_file}.npz"


def load_dataset(filename: str):
    path_to_file = f"./{filename}"
    if os.path.exists(path_to_file):
        return np.load(path_to_file)
    else:
        print(f"File doesn't exist on {path_to_file}")


def get_data_features(filename: str) -> Dict:
    dataset = load_dataset(filename)

    print("Files: ", dataset.files)

    X_train = dataset["x_train"]
    y_train = dataset["y_train"]
    X_test = dataset["x_test"]
    y_test = dataset["y_test"]
    X_train_pca = dataset["x_train_pca"]
    X_test_pca = dataset["x_test_pca"]
    eigen_faces = dataset["eigen_faces"]
    h = dataset["h"]
    w = dataset["w"]
    target_names = dataset["target_names"]
    n_classes = dataset["n_classes"]

    return {
        "data": (X_train, y_train, X_test, y_test),
        "features": (X_train_pca, X_test_pca),
        "eigenfaces": eigen_faces,
        "target_names": target_names,
        "n_classes": n_classes,
        "hw": (h, w),
    }


def get_titles(y_pred: List, y_test: List, target_names: List, i) -> str:
    pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    return "Predicted: %s\n | True:      %s" % (pred_name, true_name)


class Preprocessing:
    def __init__(self, dataset, n_samples, target_names, n_classes, hw):
        # introspect the images arrays to find the shapes (for plotting)
        self.n_samples = n_samples
        self.h, self.w = hw

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        self.X = dataset.data
        self.n_features = self.X.shape[1]

        # the label to predict is the id of the person
        self.y = dataset.target
        self.target_names = target_names
        self.n_classes = n_classes

    def scale_data(self, scaler_filename: str = "data_scaler") -> str:
        # Feature scaling
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        scaler_file_path = f"./{scaler_filename}.joblib"
        dump(scaler, scaler_file_path)

        return scaler_file_path

    def create_and_save_features(
        self,
        data_filename: str = "data",
    ) -> str:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.25, random_state=42
        )

        # Compute a PCA (eigenfaces)
        # on the face dataset (treated as unlabeled dataset):
        # unsupervised feature extraction / dimensionality reduction

        n_components = 150

        print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)

        eigenfaces = pca.components_.reshape((n_components, self.h, self.w))

        print("Projecting the input data on the eigenfaces orthonormal basis")
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        return save_dataset(
            X_train,
            y_train,
            X_test,
            y_test,
            X_train_pca,
            X_test_pca,
            eigenfaces,
            self.h,
            self.w,
            self.target_names,
            self.n_classes,
            data_filename,
        )

    def describe(self) -> None:
        print("====================================")
        print("Total dataset size")
        print("n_samples: %d" % self.n_samples)
        print("n_features: %d" % self.n_features)
        print("n_classes: %d" % self.n_classes)
