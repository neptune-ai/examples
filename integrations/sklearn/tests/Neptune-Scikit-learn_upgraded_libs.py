# Scikit-learn + Neptune

# Before you start

## Install dependencies

get_ipython().system(' pip install scikit-learn==0.24.1 neptune-client==0.5.5 neptune-sklearn==0.9.1')

get_ipython().system(' pip install --upgrade scikit-learn neptune-client neptune-sklearn')

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

# tests
run.wait()

# check tags
all_tags = ['RandomForestRegressor', 'regression']
assert set(run["sys/tags"].fetch()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, run["sys/tags"].fetch())

# check scores
assert run['rfr_summary/test/scores/explained_variance_score'].fetch() <= 1.0, 'Wrong values logged.'
assert run['rfr_summary/test/scores/max_error'].fetch() >= 0.0, 'Wrong values logged.'
assert run['rfr_summary/test/scores/mean_absolute_error'].fetch() >= 0.0, 'Wrong values logged.'
assert run['rfr_summary/test/scores/r2_score'].fetch() <= 1.0, 'Wrong values logged.'

# check parameters
for key in parameters.keys():
    assert run['rfr_summary/all_params/{}'.format(key)].fetch() == parameters[key],        'Expected: {}, Actual: {}'.format(parameters[key], run['rfr_summary/all_params/{}'.format(key)].fetch())

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

import neptune.new as neptune

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='classification-example',
                   tags=['GradientBoostingClassifier', 'classification'])

## Step 3: Log classifier summary

import neptune.new.integrations.sklearn as npt_utils

run['cls_summary'] = npt_utils.create_classifier_summary(gbc, X_train, X_test, y_train, y_test)

# tests
run.wait()

# check tags
all_tags = ['GradientBoostingClassifier', 'classification']
assert set(run["sys/tags"].fetch()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, run["sys/tags"].fetch())

# check parameters
for key in parameters.keys():
    assert run['cls_summary/all_params/{}'.format(key)].fetch() == parameters[key],        'Expected: {}, Actual: {}'.format(parameters[key], run['cls_summary/all_params/{}'.format(key)].fetch())

## Step 4: Explore Results

# Scikit-learn KMeans clustering

## Step 1: Create KMeans object and example data

parameters = {'n_init': 11,
              'max_iter': 270}

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

km = KMeans(**parameters)

X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

## Step 2: Initialize Neptune

import neptune.new as neptune

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='clustering-example',
                   tags=['KMeans', 'clustering'])

## Step 3: Log KMeans clustering summary

import neptune.new.integrations.sklearn as npt_utils

run['kmeans_summary'] = npt_utils.create_kmeans_summary(km, X, n_clusters=17)

# tests
run.wait()

# check tags
all_tags = ['KMeans', 'clustering']
assert set(run["sys/tags"].fetch()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, run["sys/tags"].fetch())

# check parameters
for key in parameters.keys():
    assert run['kmeans_summary/all_params/{}'.format(key)].fetch() == parameters[key],        'Expected: {}, Actual: {}'.format(parameters[key], run['kmeans_summary/all_params/{}'.format(key)].fetch())

## Explore Results

# Other logging options

## Before you start: create and fit gradient boosting classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfc.fit(X_train, y_train)

## Import sklearn integration that will be used below, and create new run

import neptune.new.integrations.sklearn as npt_utils

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='other-options')

## Log estimator parameters

run['estimator/parameters'] = npt_utils.get_estimator_params(rfc)

## Log model

run['estimator/pickled-model'] = npt_utils.get_pickled_model(rfc)

## Log confusion matrix

run['confusion-matrix'] = npt_utils.create_confusion_matrix_chart(rfc, X_train, X_test, y_train, y_test)