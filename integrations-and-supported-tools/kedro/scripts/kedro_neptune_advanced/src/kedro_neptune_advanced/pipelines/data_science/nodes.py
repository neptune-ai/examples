# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

from typing import Any, Dict

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
from scikitplot.metrics import plot_precision_recall_curve, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def train_rf_model(train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]):
    """Node for training Random Forest model"""
    max_depth = parameters["rf_max_depth"]
    n_estimators = parameters["rf_n_estimators"]
    max_features = parameters["rf_max_features"]

    clf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, max_features=max_features
    )
    clf.fit(train_x, train_y.idxmax(axis=1))

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
    """Node for evaluating Random Forest and MLP models and creating ROC and Precision-Recall Curves"""

    for name, y_pred in predictions.items():
        y_true = test_y.to_numpy().argmax(axis=1)
        y_pred = np.array(y_pred)

        accuracy = accuracy_score(y_true, y_pred.argmax(axis=1).ravel())
        neptune_run[f"nodes/evaluate_models/metrics/accuracy_{name}"] = accuracy

        fig, ax = plt.subplots()
        plot_roc_curve(test_y.idxmax(axis=1), y_pred, ax=ax, title=f"ROC curve {name}")
        neptune_run["nodes/evaluate_models/plots/plot_roc_curve"].log(fig)

        fig, ax = plt.subplots()
        plot_precision_recall_curve(test_y.idxmax(axis=1), y_pred, ax=ax, title=f"PR curve {name}")
        neptune_run["nodes/evaluate_models/plots/plot_precision_recall_curve"].log(fig)


def ensemble_models(
    predictions: dict, test_y: pd.DataFrame, neptune_run: neptune.handler.Handler
) -> None:
    """Node for averaging predictions of Random Forest and MLP models"""
    y_true = test_y.to_numpy().argmax(axis=1)
    y_pred_averaged = np.stack(predictions.values()).mean(axis=0)

    accuracy = accuracy_score(y_true, y_pred_averaged.argmax(axis=1).ravel())
    neptune_run[f"nodes/ensemble_models/metrics/accuracy_ensemble"] = accuracy
