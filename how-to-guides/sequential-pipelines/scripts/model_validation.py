import matplotlib
import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.exceptions import ModelNotFound, ModelVersionNotFound
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from utils import *

matplotlib.use("Agg")

run = neptune.init_run(
    monitoring_namespace="monitoring/validation",
)

model_name = "pickled_model"
dataset_name = "features"
nrows = 1000

# (Neptune) Get dataset features from preprocessing stage
run["preprocessing/dataset"][dataset_name].download()

# (Neptune) Get latest model from training stage
model_key = "PIPELINES"
project_key = run["sys/id"].fetch().split("-")[0]

try:
    # (Neptune) Resume model
    model = neptune.init_model(
        with_id=f"{project_key}-{model_key}",  # Your model ID here
    )
    latest_model_version_id = model.fetch_model_versions_table().to_pandas()["sys/id"][0]


except ModelNotFound:
    print(
        f"The model with the provided key `{model_key}` doesn't exist in the `{project_key}` project."
    )

try:
    # (Neptune) Resume model version created in the training stage
    model_version = neptune.init_model_version(with_id=latest_model_version_id)
except ModelVersionNotFound:
    print(
        f"The model version with the ID `{latest_model_version_id}` doesn't exist in the `{project_key}-{model_key}` model."
    )

# (Neptune) Get model weights from training stage
model_version["model"][model_name].download()

# Load model and dataset
clf = load_model(f"{model_name}.pkl")
eval_data = load_dataset(f"{dataset_name}.npz")

# Get dataset details
target_names = eval(np.array_str(eval_data["target_names"]))
h = eval_data["h"]
w = eval_data["w"]

# Get train set features
X_train_pca = eval_data["x_train_pca"]
y_train = eval_data["y_train"]

# Get test set features
X_test = eval_data["x_test"]
y_test = eval_data["y_test"]
X_test_pca = eval_data["x_test_pca"]
eigen_faces = eval_data["eigen_faces"]

# Predict on test set
y_pred = clf.predict(X_test_pca)

# Get predicted prediction titles
prediction_titles = [get_titles(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

# Get eigenface titles
eigenface_titles = ["eigenface %d" % i for i in range(eigen_faces.shape[0])]

# (Neptune) Set up "validation" namespace inside the run.
# This will be the base namespace where all the validation metadata is logged.
validation_handler = run["validation"]

for i, image in enumerate(X_test):
    fig = plt.figure()
    img = plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

    # (Neptune) Log predicted images
    validation_handler["images/predicted_images"].append(fig, description=prediction_titles[i])

    plt.close()

for i, image in enumerate(eigen_faces):
    fig = plt.figure()
    img = plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

    # (Neptune) Log eigen face images
    validation_handler["images/eigen_faces"].append(fig, description=eigenface_titles[i])

    plt.close()

# Confusion matrix
print(classification_report(y_test, y_pred, target_names=target_names))
confusion_matrix = ConfusionMatrixDisplay.from_estimator(
    clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
)
confusion_matrix_filename = "confusion_matrix"
confusion_matrix.figure_.savefig(confusion_matrix_filename)

# (Neptune) Log validation scores
validation_handler["metrics"] = {
    "preds": npt_utils.get_test_preds(clf, X_test_pca, y_test, y_pred=y_pred, nrows=nrows),
    "preds_proba": npt_utils.get_test_preds_proba(clf, X_test_pca, nrows=nrows),
    "scores": npt_utils.get_scores(clf, X_test_pca, y_test, y_pred=y_pred),
}

# (Neptune) Log validation diagnostics charts
validation_handler["metrics/diagnostics_charts"] = {
    "confusion_matrix": File(f"{confusion_matrix_filename}.png"),
    "classification_report": npt_utils.create_classification_report_chart(
        clf, X_train_pca, X_test_pca, y_train, y_test
    ),
    "ROC_AUC": npt_utils.create_roc_auc_chart(clf, X_train_pca, X_test_pca, y_train, y_test),
    "class_prediction_error": npt_utils.create_class_prediction_error_chart(
        clf, X_train_pca, X_test_pca, y_train, y_test
    ),
}

# (Neptune) Log metrics to model registry
run.wait()
model_score = validation_handler["metrics/scores"].fetch()
model_version["metrics/validation/scores"] = model_score

# (Neptune) Move model to staging
SCORE_THRESHOLD = 0.50
if model_score["class_0"]["fbeta_score"] > SCORE_THRESHOLD:
    model_version.change_stage("staging")
else:
    model_version.change_stage("archived")
