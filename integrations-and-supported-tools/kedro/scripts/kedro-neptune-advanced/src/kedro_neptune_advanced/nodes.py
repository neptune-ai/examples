"""
This is a boilerplate pipeline
generated using Kedro 0.18.4
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
from scikitplot.metrics import plot_precision_recall, plot_roc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    data_train = data.sample(
        frac=parameters["train_fraction"], random_state=parameters["random_state"]
    )
    data_test = data.drop(data_train.index)

    X_train = data_train.drop(columns=parameters["target_column"])
    X_test = data_test.drop(columns=parameters["target_column"])
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test


def train_rf_model(train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]):

    max_depth = parameters["rf_max_depth"]
    n_estimators = parameters["rf_n_estimators"]
    max_features = parameters["rf_max_features"]

    clf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        max_features=max_features,
    )
    clf.fit(train_x, train_y)

    return clf


def train_mlp_model(train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]):
    """Node for training MLP model"""
    alpha = parameters["mlp_alpha"]
    max_iter = parameters["mlp_max_iter"]

    clf = MLPClassifier(alpha=alpha, max_iter=max_iter)
    clf.fit(train_x, train_y)

    return clf


def get_predictions(
    rf_model: RandomForestClassifier, mlp_model: MLPClassifier, test_x: pd.DataFrame
) -> Dict[str, Any]:
    """Node for making predictions given a pre-trained model and a test set."""
    predictions = {}
    for name, model in zip(["rf", "mlp"], [rf_model, mlp_model]):
        y_pred = model.predict_proba(test_x).tolist()
        predictions[name] = y_pred

    return predictions


def evaluate_models(predictions: dict, test_y: pd.DataFrame, neptune_run: neptune.handler.Handler):
    """Node for
    - evaluating Random Forest and MLP models
    - creating ROC and Precision-Recall Curves
    """

    for name, y_pred_proba in predictions.items():

        y_true = test_y.to_numpy()
        y_pred_proba = np.array(y_pred_proba)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred = np.where(y_pred == 0, "setosa", np.where(y_pred == 1, "versicolor", "virginica"))

        accuracy = accuracy_score(y_true, y_pred)
        neptune_run[f"nodes/evaluate_models/metrics/accuracy_{name}"] = accuracy

        fig, ax = plt.subplots()
        plot_roc(test_y, y_pred_proba, ax=ax, title=f"ROC curve {name}")
        neptune_run["nodes/evaluate_models/plots/plot_roc_curve"].append(fig)

        fig, ax = plt.subplots()
        plot_precision_recall(test_y, y_pred_proba, ax=ax, title=f"PR curve {name}")
        neptune_run["nodes/evaluate_models/plots/plot_precision_recall_curve"].append(fig)
