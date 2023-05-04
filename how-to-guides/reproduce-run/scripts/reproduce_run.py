import os

import neptune
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# (Neptune) Setting up credentials env variables
os.environ["NEPTUNE_PROJECT"] = "common/showroom"  # You can replace this with your own project
os.environ[
    "NEPTUNE_API_TOKEN"
] = neptune.ANONYMOUS_API_TOKEN  # You can replace this with your own token

################################
# (Neptune) Step 1: Get run ID #
################################
# Fetch only inactive runs with tags "showcase-run", "reproduce" and "Basic script" from project
with neptune.init_project(mode="read-only") as project:
    runs_table_df = project.fetch_runs_table(
        state="inactive", tag=["showcase-run", "reproduce", "Basic script"]
    ).to_pandas()

# Extract the last successful run's id
old_run_id = runs_table_df[runs_table_df["sys/failed"] == False]["sys/id"].values[0]

print(f"old_run_id = {old_run_id}")

#############################################################################
# (Neptune) Step 2: Resume old run and fetch relevant metadata from Neptune #
#############################################################################
# Use the `neptune.init_run()` method to:
# - Re-open an existing run using the ID you got from the previous step
# - Re-open it in the `read-only` mode so that metadata logged to the old run is not accidentally changed
old_run = neptune.init_run(
    with_id=old_run_id,
    mode="read-only",
)

# Fetch hyperparameters
old_run_params = old_run["config/params"].fetch()

# Fetch dataset path
dataset_path = old_run["config/dataset/path"].fetch()

######################################
# (Neptune) Step 3: Create a new run #
######################################
# Create a new Neptune run that will be used to log metadata in the re-run session.
new_run = neptune.init_run(tags=["reproduce", "new-run"])

#####################################################################################
# (Neptune) Step 4: Log hyperparameters and dataset details from old run to new run #
#####################################################################################
# Now you can continue working and logging metadata to a brand new run.
# You can log metadata using the Neptune API Client. For details, see [What you can log and display](https://docs.neptune.ai/logging/what_you_can_log).

new_run["config/params"] = old_run_params
new_run["config/dataset/path"] = dataset_path

# Load dataset and model
# Dataset
data_tfms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

trainset = datasets.CIFAR10(dataset_path, transform=data_tfms["train"], download=True)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=old_run_params["bs"], shuffle=True, num_workers=0
)


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


model = BaseModel(
    old_run_params["input_sz"],
    old_run_params["input_sz"],
    old_run_params["n_classes"],
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=old_run_params["lr"])

############################################
# (Neptune) Step 5: Log losses and metrics #
############################################
for i, (x, y) in enumerate(trainloader, 0):
    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    new_run["training/batch/loss"].append(loss)

    new_run["training/batch/acc"].append(acc)

    loss.backward()
    optimizer.step()
