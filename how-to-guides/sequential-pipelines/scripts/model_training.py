import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.exceptions import NeptuneModelKeyAlreadyExistsError
from neptune.utils import stringify_unsupported
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from utils import get_data_features

# (Neptune) Create a new run
run = neptune.init_run(
    monitoring_namespace="monitoring/training",
)

# (Neptune) Fetch features from preprocessing stage
run["preprocessing/dataset/features"].download()

# (Neptune) Set up "training" namespace inside the run.
# This will be the base namespace where all the training metadata is logged.
training_handler = run["training"]

# Get features
dataset = get_data_features("features.npz")
X_train, y_train, X_test, y_test = dataset["data"]
X_train_pca, X_test_pca = dataset["features"]

# Train a SVM classification model
print("Fitting the classifier to the training set")
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}

# Train a SVM classification model
clf = RandomizedSearchCV(
    SVC(kernel="rbf", class_weight="balanced", probability=True), param_grid, n_iter=10
)
clf = clf.fit(X_train_pca, y_train)

print("Best model found by grid search:")
print(clf.best_estimator_)

# (Neptune) Log model params
training_handler["params"] = stringify_unsupported(npt_utils.get_estimator_params(clf))

# (Neptune) Log model scores
training_handler["metrics/scores"] = npt_utils.get_scores(clf, X_train_pca, y_train)

# (Neptune) Log pickled model
model_name = "pickled_model"
training_handler["model"][model_name] = npt_utils.get_pickled_model(clf)

# (Neptune) Initializing a Model and Model version
model_key = "PIPELINES"
project_key = run["sys/id"].fetch().split("-")[0]

try:
    model = neptune.init_model(key=model_key)
    model.wait()
    print("Creating a new model version...")
    model_version = neptune.init_model_version(model=f"{project_key}-{model_key}")

except NeptuneModelKeyAlreadyExistsError:
    print(f"A model with the provided key {model_key} already exists in this project.")
    print("Creating a new model version...")
    model_version = neptune.init_model_version(
        model=f"{project_key}-{model_key}",
    )

# (Neptune) Log model version details to run
model_version.wait()
training_handler["model/model_version/id"] = model_version["sys/id"].fetch()
training_handler["model/model_version/model_id"] = model_version["sys/model_id"].fetch()
training_handler["model/model_version/url"] = model_version.get_url()

# (Neptune) Log run details
model_version["run/id"] = run["sys/id"].fetch()
model_version["run/name"] = run["sys/name"].fetch()
model_version["run/url"] = run.get_url()

# (Neptune) Log training scores from run
run.wait()
model_scores = training_handler["metrics/scores"].fetch()
model_version["metrics/training/scores"] = model_scores

# (Neptune) Download pickled model from run
training_handler["model"][model_name].download()

# (Neptune) Upload pickled model to model registry
model_version["model"][model_name].upload("pickled_model.pkl")
