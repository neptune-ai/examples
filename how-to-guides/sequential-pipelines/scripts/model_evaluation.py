import neptune.new as neptune
from neptune.new.types import File
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from utils import *

run = neptune.init_run(project="common/showroom", api_token=neptune.ANONYMOUS_API_TOKEN)

# (Neptune) Get model weights and data
run["training/pickled_model"].download()
run["prepocessing/data"].download()

# Load model and data
clf = load_model("pickled_model.joblib")
eval_data = load_dataset("data.npz")

# Get dataset details
target_names = eval_data["target_names"]
h = eval_data["h"]
w = eval_data["w"]

# Get test set features
X_test = eval_data["x_test"]
y_test = eval_data["y_test"]
X_test_pca = eval_data["x_test_pca"]
eigen_faces = eval_data["eigen_faces"]

# Predict on test set
y_pred = clf.predict(X_test_pca)

# Print confusion matrix
print(classification_report(y_test, y_pred, target_names=target_names))
confusion_matrix = ConfusionMatrixDisplay.from_estimator(
    clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
)

# Get predicted prediction titles
prediction_titles = [get_titles(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

# Get eigenface titles
eigenface_titles = ["eigenface %d" % i for i in range(eigen_faces.shape[0])]

# (Neptune) Set basenamespace
handler_run = run["eval"]

# (Neptune) Log confusion matrix
handler_run["confusion_matix"].upload(confusion_matrix.figure_)

for i, image in enumerate(X_test):
    fig = plt.figure()
    img = plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

    # (Neptune) Log predicted images
    handler_run["images/predicted_images"].append(fig, description=prediction_titles[i])

    plt.close()

for i, image in enumerate(eigen_faces):
    fig = plt.figure()
    img = plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

    # (Neptune) Log eigen face images
    handler_run["images/eigen_faces"].append(fig, description=eigenface_titles[i])

    plt.close()
