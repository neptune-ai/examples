import os
import uuid
from functools import reduce

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from neptune.utils import stringify_unsupported
from torchvision import datasets, transforms
from tqdm.auto import trange

# Log to public project
os.environ["NEPTUNE_API_TOKEN"] = neptune.ANONYMOUS_API_TOKEN
os.environ["NEPTUNE_PROJECT"] = "common/hpo"

## **To Log to your own project instead**
# Uncomment the code block below:

# from getpass import getpass
# os.environ["NEPTUNE_API_TOKEN"]=getpass("Enter your Neptune API token: ")
# os.environ["NEPTUNE_PROJECT"]="workspace-name/project-name",  # replace with your own

# Create a sweep identifier
sweep_id = str(uuid.uuid4())

# Hyperparameters
parameters = {
    "batch_size": 256,
    "epochs": 2,
    "input_size": (3, 32, 32),
    "n_classes": 10,
    "dataset_size": 1000,
    "model_filename": "basemodel",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

input_size = reduce(lambda x, y: x * y, parameters["input_size"])

## Hyperparameter search space
learning_rates = [0.01, 0.05, 0.1]  # learning rate choices


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

# Create a sweep level run
sweep_run = neptune.init_run(
    tags=["script", "sweep-level"],
)

# Add sweep_id to the sweep run
sweep_run["sys/group_tags"].add(sweep_id)

# Training loop
for i, lr in enumerate(learning_rates):
    # Create trial-level run
    with neptune.init_run(
        name=f"trial-{i}",
        tags=[
            "script",
            "trial-level",
        ],  # to indicate that the run only contains results from a single trial
    ) as trial_run:
        # Add sweep_id to the trial-level run
        trial_run["sys/group_tags"].add(sweep_id)

        # Log hyperparameters
        trial_run["params"] = stringify_unsupported(parameters)
        trial_run["params/lr"] = lr

        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Initialize fields for best values across all trials
        best_loss = None

        for _ in trange(parameters["epochs"]):
            for x, y in trainloader:
                x, y = x.to(parameters["device"]), y.to(parameters["device"])
                optimizer.zero_grad()
                outputs = model.forward(x)
                loss = criterion(outputs, y)

                _, preds = torch.max(outputs, 1)
                acc = (torch.sum(preds == y.data)) / len(x)

                # Log trial metrics
                trial_run["metrics/batch/loss"].append(loss)
                trial_run["metrics/batch/acc"].append(acc)

                # Log best values across all trials to sweep-level run
                if best_loss is None or loss < best_loss:
                    sweep_run["best/trial"] = i
                    sweep_run["best/metrics/loss"] = best_loss = loss
                    sweep_run["best/metrics/acc"] = acc
                    sweep_run["best/params"] = stringify_unsupported(parameters)
                    sweep_run["best/params/lr"] = lr

                loss.backward()
                optimizer.step()

# Stop sweep-level run
sweep_run.stop()
