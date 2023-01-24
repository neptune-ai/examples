import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.exceptions import NeptuneException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from utils import get_data_features

# (Neptune) Create a new run
run = neptune.init_run(
    monitoring_namespace="monitoring/training",
)

# (Neptune) Fetch features from preprocessing stage
run["preprocessing/dataset/features"].download()

# (Neptune) Set basenamespace
handler_run = run["training"]

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
handler_run["params"] = npt_utils.get_estimator_params(clf)

# (Neptune) Log model scores
handler_run["metrics/scores"] = npt_utils.get_scores(clf, X_train_pca, y_train)

# (Neptune) Log pickled model
model_name = "pickled_model"
handler_run[f"model/{model_name}"] = npt_utils.get_pickled_model(clf)

# (Neptune) Initializing a Model and Model version
model_key = "PIPELINES"
project_key = run["sys/id"].fetch().split("-")[0]

try:
    model = neptune.init_model(key=model_key)

    print("Creating a new model version...")
    model_version = neptune.init_model_version(model=f"{project_key}-{model_key}")

except NeptuneException:
    print(f"A model with the provided key {model_key} already exists in this project.")
    print("Creating a new model version...")
    model_version = neptune.init_model_version(
        model=f"{project_key}-{model_key}",
    )

# (Neptune) Log model version details to run
handler_run["model/model_version/id"] = model_version["sys/id"].fetch()
handler_run["model/model_version/model_id"] = model_version["sys/model_id"].fetch()
handler_run["model/model_version/url"] = model_version.get_url()

# (Neptune) Log run details
model_version["run/id"] = run["sys/id"].fetch()
model_version["run/name"] = run["sys/name"].fetch()
model_version["run/url"] = run.get_url()

# (Neptune) Log training scores from run
run.wait()
model_scores = run["training/metrics/scores"].fetch()
model_version["metrics/training/scores"] = model_scores

# (Neptune) Download pickled model from Run
run[f"training/model/{model_name}"].download()

# (Neptune) Upload pickled model to Model registry
model_version[f"model/{model_name}"].upload(f"pickled_model.pkl")
