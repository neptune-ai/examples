import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Hyperparameters
parameters = {
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "model_filename": "basemodel",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Hyperparameter search space
learning_rates = [1e-4, 1e-3, 1e-2]  # learning rate choices

# Model
class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


model = BaseModel(parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]).to(
    parameters["device"]
)

criterion = nn.CrossEntropyLoss()

# Dataset
data_dir = "data/CIFAR10"
compressed_ds = "./data/CIFAR10/cifar-10-python.tar.gz"
data_tfms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=parameters["bs"], shuffle=True)


# Log metadata from each trial into separate run
for (i, lr) in enumerate(learning_rates):
    # (Neptune) Create a run
    run = neptune.init_run(
        api_token=neptune.ANONYMOUS_API_TOKEN,
        project="common/pytorch-integration",
        name=f"trial-{i}",
    )

    # (Neptune) Log hyperparameters
    run["parms"] = parameters
    run["parms/lr"] = lr

    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i, (x, y) in enumerate(trainloader, 0):

        x, y = x.to(parameters["device"]), y.to(parameters["device"])
        optimizer.zero_grad()
        outputs = model.forward(x)
        loss = criterion(outputs, y)

        _, preds = torch.max(outputs, 1)
        acc = (torch.sum(preds == y.data)) / len(x)

        # (Neptune) Log losses and metrics
        run["training/batch/loss"].log(loss)
        run["training/batch/acc"].log(acc)

        loss.backward()
        optimizer.step()

    # (Neptune) Wait for all the tracking calls to finish.
    run.wait()
    # (Neptune) Stop logging
    run.stop()
