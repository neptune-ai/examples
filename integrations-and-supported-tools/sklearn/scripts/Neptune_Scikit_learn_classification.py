import matplotlib.pyplot as plt
import neptune
import neptune.integrations.sklearn as npt_utils
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# To fix the random RuntimeError: main thread is not in main loop error in Windows running python 3.8
plt.switch_backend("agg")

run = neptune.init_run(
    project="common/sklearn-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    name="classification-example",
    tags=["GradientBoostingClassifier", "classification"],
)

parameters = {
    "n_estimators": 120,
    "learning_rate": 0.12,
    "min_samples_split": 3,
    "min_samples_leaf": 2,
}

gbc = GradientBoostingClassifier(**parameters)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

gbc.fit(X_train, y_train)

run["cls_summary"] = npt_utils.create_classifier_summary(gbc, X_train, X_test, y_train, y_test)
