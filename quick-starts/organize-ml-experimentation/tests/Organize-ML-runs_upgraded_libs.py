# Organize ML runs

# Setup

get_ipython().system(' pip install --quiet neptune-client==0.5.5 scikit-learn==0.23.1')

get_ipython().system(' pip install --upgrade --quiet neptune-client scikit-learn')

# Step 1: Create a basic training script

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.4, random_state=1234)

params = {'n_estimators': 10,
          'max_depth': 3,
          'min_samples_leaf': 1,
          'min_samples_split': 2,
          'max_features': 3,
          }

clf = RandomForestClassifier(**params)

clf.fit(X_train, y_train)
y_train_pred = clf.predict_proba(X_train)
y_test_pred = clf.predict_proba(X_test)

train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average='macro')
test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average='macro')
print(f'Train f1:{train_f1} | Test f1:{test_f1}')

# Step 2: Initialize Neptune and create new run

import neptune.new as neptune

run = neptune.init(project='common/quickstarts',
                   api_token='ANONYMOUS')

# Step 3: Save parameters

run['parameters'] = params

# Step 4. Add tags to organize things

run["sys/tags"].add(['run-organization', 'me'])

# Step 5. Add logging of train and evaluation metrics

run['train/f1'] = train_f1
run['test/f1'] = test_f1

# Step 6. Execute a few runs with different parameters

# tests
run.wait()

# check metrics
assert isinstance(run['train/f1'].fetch(), float), 'Incorrect metric type'
assert isinstance(run['test/f1'].fetch(), float), 'Incorrect metric type'