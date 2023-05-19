import neptune
import numpy as np
import torch
import torch.nn.functional as F
from neptune.types import File
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, NeptuneLogger
from torch import nn

# Define hyper-parameters
params = {
    "batch_size": 2,
    "lr": 0.1,
    "max_epochs": 10,
}

# Load data
mnist = fetch_openml("mnist_784", as_frame=False, cache=False)

# Preprocess data
X = mnist.data.astype("float32")
y = mnist.target.astype("int64")
X /= 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build a neural network with PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"
mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim / 8)
output_dim = len(np.unique(mnist.target))


class ClassifierModule(nn.Module):
    def __init__(
        self,
        input_dim=mnist_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


# (Neptune) Initialize Neptune run
run = neptune.init_run(api_token=neptune.ANONYMOUS_API_TOKEN, project="common/skorch-integration")
# (Neptune) Create NeptuneLogger
neptune_logger = NeptuneLogger(run, close_after_train=False)

# Initialize checkpoint callback
checkpoint_dirname = "./checkpoints"
checkpoint = Checkpoint(dirname=checkpoint_dirname)

# Initialize a trainer and pass neptune_logger
net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=params["max_epochs"],
    lr=params["lr"],
    device=device,
    callbacks=[neptune_logger, checkpoint],
)

# Train the model and log metadata to the Neptune run
net.fit(X_train, y_train)

# (Neptune) Log model weights
neptune_logger.run["training/model/checkpoints"].upload_files(checkpoint_dirname)

# (Neptune) Log test score
y_pred = net.predict(X_test)
neptune_logger.run["training/test/acc"] = accuracy_score(y_test, y_pred)

# (Neptune) Log misclassified images
error_mask = y_pred != y_test
for x, y_hat, y in zip(X_test[error_mask], y_pred[error_mask], y_test[error_mask]):
    x_reshaped = x.reshape(28, 28)
    neptune_logger.run["training/test/misclassified_images"].append(
        File.as_image(x_reshaped), description=f"y_pred={y_hat}, y_true={y}"
    )
