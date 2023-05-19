import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Step 1: Get Run ID
## Get project
project = neptune.init_project(
    project="common/showroom", api_token=neptune.ANONYMOUS_API_TOKEN, mode="read-only"
)

## Fetch only inactive runs with tag "showcase-run"
runs_table_df = project.fetch_runs_table(
    state="inactive", tag=["showcase-run"], columns=["sys/failed"]
).to_pandas()

## Extract the last failed run's id
failed_run_id = runs_table_df[runs_table_df["sys/failed"] == True]["sys/id"].values[0]

print("Failed_run_id = ", failed_run_id)

# Step 2: Resume failed run
failed_run = neptune.init_run(
    project="common/showroom",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    with_id=failed_run_id,
    mode="read-only",
)

## Step 3: Use the fetch() method to retrieve relevant metadata
## Fetch hyperparameters
failed_run_params = failed_run["config/hyperparameters"].fetch()
## Fetch dataset path
dataset_path = failed_run["dataset/path"].fetch()


# Step 4: Create a new run
## Create a new Neptune run that will be used to log metadata in the re-run session.
new_run = neptune.init_run(
    project="common/showroom",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["re-run", "successful training"],
)

# Step 5: Log Hyperparameters and Dataset details from failed run to new run
## Now you can continue working and logging metadata to a brand new Run.
## You can log metadata using the Neptune API Client
new_run["config/hyperparameters"] = failed_run_params
new_run["dataset/path"] = dataset_path

## Load Dataset and Model
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
    trainset, batch_size=failed_run_params["bs"], shuffle=True, num_workers=0
)


## Model
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
    failed_run_params["input_sz"],
    failed_run_params["input_sz"],
    failed_run_params["n_classes"],
).to(failed_run_params["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=failed_run_params["lr"])

## Log losses and metrics
for i, (x, y) in enumerate(trainloader, 0):
    x, y = x.to(failed_run_params["device"]), y.to(failed_run_params["device"])
    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    new_run["training/batch/loss"].append(loss)

    new_run["training/batch/acc"].append(acc)

    loss.backward()
    optimizer.step()
