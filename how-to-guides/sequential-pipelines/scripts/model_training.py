#%%
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from utils import get_data_features, save_model

# %%
# Get features
dataset = get_data_features("data.npz")
X_train, y_train, X_test, y_test = dataset["data"]
X_train_pca, X_test_pca = dataset["features"]

# %%
# Train a SVM classification model

print("Fitting the classifier to the training set")
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}
clf = RandomizedSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

save_model(clf, "pickled_model")


# %%
