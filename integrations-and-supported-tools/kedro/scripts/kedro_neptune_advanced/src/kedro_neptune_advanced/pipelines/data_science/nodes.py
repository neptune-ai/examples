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

import logging
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
from kedro.extras.datasets.pickle import PickleDataSet
from scikitplot.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Dict


def train_rf_model(
        train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
):
    X = train_x.to_numpy()
    y = train_y.to_numpy()

    max_depth = parameters["rf_max_depth"]
    n_estimators = parameters["rf_n_estimators"]
    max_features = parameters["rf_max_features"]

    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)
    clf.fit(X, y)

    model_dataset = PickleDataSet(filepath="data/06_models/rf_model.pkl", backend="pickle")
    model_dataset.save(clf)


def train_tree_model(
        train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
):
    X = train_x.to_numpy()
    y = train_y.to_numpy()

    max_depth = parameters["tree_max_depth"]

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)

    model_dataset = PickleDataSet(filepath="data/06_models/tree_model.pkl", backend="pickle")
    model_dataset.save(clf)


def train_mlp_model(
        train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
):
    X = train_x.to_numpy()
    y = train_y.to_numpy()

    alpha = parameters["mlp_alpha"]
    max_iter = parameters["mlp_max_iter"]

    clf = MLPClassifier(alpha=alpha, max_iter=max_iter)
    clf.fit(X, y)

    model_dataset = PickleDataSet(filepath="data/06_models/mlp_model.pkl", backend="pickle")
    model_dataset.save(clf)


def evaluate(rf_model: RandomForestClassifier,
             tree_model: DecisionTreeClassifier,
             mlp_model: MLPClassifier,
             test_x: pd.DataFrame, test_y: pd.DataFrame,
             neptune_run: neptune.run.Handler) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set."""
    X = test_x.to_numpy()
    y_true = test_y.to_numpy()

    for name, model in zip(['rf', 'tree', 'mlp'], [rf_model, tree_model, mlp_model]):
        y_pred = model.predict(X)
        y_true = y_true.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        neptune_run[f'nodes/evaluate/metrics/accuracy_{name}'] = accuracy

        fig, ax = plt.subplots()
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        fig.title = name
        neptune_run['nodes/evaluate/plots/confusion_matrix'].log(fig)
