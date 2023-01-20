# %%
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from utils import get_data_features, save_model

# (Neptune) Create a new run
run = neptune.init_run(
    project="common/showroom",
    api_token=neptune.ANONYMOUS_API_TOKEN,
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
handler_run["model/pickled_model"] = npt_utils.get_pickled_model(clf)
