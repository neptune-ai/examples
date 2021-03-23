# Scikit-learn + Neptune

# Before you start

## Install dependencies

# Scikit-learn regression

## Step 1: Create and fit random forest regressor

parameters = {'n_estimators': 70,
              'max_depth': 7,
              'min_samples_split': 3}

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

rfr = RandomForestRegressor(**parameters)

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfr.fit(X_train, y_train)

## Step 2: Initialize Neptune

import neptune.new as neptune

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='regression-example',
                   tags=['RandomForestRegressor', 'regression'])

## Step 3: Log regressor summary

import neptune.new.integrations.sklearn as npt_utils

run['rfr_summary'] = npt_utils.create_regressor_summary(rfr, X_train, X_test, y_train, y_test)

## Step 4: Explore results

# Scikit-learn classification

## Step 1: Create and fit gradient boosting classifier

parameters = {'n_estimators': 120,
              'learning_rate': 0.12,
              'min_samples_split': 3,
              'min_samples_leaf': 2}

from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

gbc = GradientBoostingClassifier(**parameters)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

gbc.fit(X_train, y_train)

## Step 2: Initialize Neptune

import neptune

neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')

## Step 3: Create a run

neptune.create_experiment(params=parameters,
                          name='classification-example',
                          tags=['GradientBoostingClassifier', 'classification'])

## Step 4: Log classifier summary

from neptunecontrib.monitoring.sklearn import log_classifier_summary

log_classifier_summary(gbc, X_train, X_test, y_train, y_test)

## Step 5: Stop Neptune run after logging summary

neptune.stop()

## Explore Results

# Scikit-learn KMeans clustering

## Step 1: Create KMeans object and example data

parameters = {'n_init': 11,
              'max_iter': 270}

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

km = KMeans(**parameters)

X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

## Step 2: Initialize Neptune

import neptune

neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')

## Step 3: Create a run

neptune.create_experiment(params=parameters,
                          name='clustering-example',
                          tags=['KMeans', 'clustering'])

## Step 4: Log KMeans clustering summary

from neptunecontrib.monitoring.sklearn import log_kmeans_clustering_summary

log_kmeans_clustering_summary(km, X, n_clusters=17)

## Step 5: Stop Neptune run after logging summary

neptune.stop()

## Explore Results

# Other logging options

## Before you start: create and fit gradient boosting classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfc.fit(X_train, y_train)

## Log estimator parameters

from neptunecontrib.monitoring.sklearn import log_estimator_params

neptune.create_experiment(name='estimator-params')

log_estimator_params(rfc) # log estimator parameters here

neptune.stop()

## Log model

from neptunecontrib.monitoring.sklearn import log_pickled_model

neptune.create_experiment(name='pickled-model')

log_pickled_model(rfc, 'my_model') # log pickled model parameters here.
                                   # path to file in the Neptune artifacts is ``model/<my_model>``.

neptune.stop()

## Log confusion matrix

from neptunecontrib.monitoring.sklearn import log_confusion_matrix_chart

neptune.create_experiment(name='confusion-matrix-chart')

log_confusion_matrix_chart(rfc, X_train, X_test, y_train, y_test) # log confusion matrix chart

neptune.stop()