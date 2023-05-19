from functools import reduce

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from neptune.utils import stringify_unsupported
from torchvision import datasets, transforms
from tqdm.auto import trange

# Hyperparameters
parameters = {
    "batch_size": 128,
    "epochs": 1,
    "input_size": (3, 32, 32),
    "n_classes": 10,
    "dataset_size": 1000,
    "model_filename": "basemodel",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

input_size = reduce(lambda x, y: x * y, parameters["input_size"])

# Hyperparameter search space
learning_rates = [1e-4, 1e-3, 1e-2]  # learning rate choices


# Model
class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.input_size = input_size

    def forward(self, input):
        x = input.view(-1, self.input_size)
        return self.main(x)


model = BaseModel(
    input_size,
    input_size,
    parameters["n_classes"],
).to(parameters["device"])


criterion = nn.CrossEntropyLoss()


# Dataset
data_tfms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
}

trainset = datasets.FakeData(
    size=parameters["dataset_size"],
    image_size=parameters["input_size"],
    num_classes=parameters["n_classes"],
    transform=data_tfms["train"],
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=parameters["batch_size"], shuffle=True, num_workers=0
)


# Log metadata from each trial into separate run
for i, lr in enumerate(learning_rates):
    # (Neptune) Create a run
    run = neptune.init_run(
        api_token=neptune.ANONYMOUS_API_TOKEN,
        project="common/pytorch-integration",
        name=f"trial-{i}",
        tags=["trial-level"],
    )

    # (Neptune) Log hyperparameters
    run["parms"] = stringify_unsupported(parameters)
    run["parms/lr"] = lr

    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in trange(parameters["epochs"]):
        for i, (x, y) in enumerate(trainloader, 0):
            x, y = x.to(parameters["device"]), y.to(parameters["device"])
            optimizer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs, y)

            _, preds = torch.max(outputs, 1)
            acc = (torch.sum(preds == y.data)) / len(x)

            # (Neptune) Log losses and metrics
            run["training/batch/loss"].append(loss)
            run["training/batch/acc"].append(acc)

            loss.backward()
            optimizer.step()
