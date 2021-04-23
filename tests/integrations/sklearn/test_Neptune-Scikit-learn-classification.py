from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='classification-example',
                   tags=['GradientBoostingClassifier', 'classification'])

parameters = {'n_estimators': 120,
              'learning_rate': 0.12,
              'min_samples_split': 3,
              'min_samples_leaf': 2}

gbc = GradientBoostingClassifier(**parameters)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

gbc.fit(X_train, y_train)

run['cls_summary'] = npt_utils.create_classifier_summary(gbc, X_train, X_test, y_train, y_test)

# tests
run.wait()

# check tags
all_tags = ['GradientBoostingClassifier', 'classification']
assert set(run["sys/tags"].fetch()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, run["sys/tags"].fetch())

# check parameters
for key in parameters.keys():
    assert run['cls_summary/all_params/{}'.format(key)].fetch() == parameters[key],        'Expected: {}, Actual: {}'.format(parameters[key], run['cls_summary/all_params/{}'.format(key)].fetch())
