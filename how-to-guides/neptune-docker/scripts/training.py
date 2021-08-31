import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import neptune.new as neptune
import os


# read docker secret as environment
with open(os.environ['NEPTUNE_API_TOKEN']) as file:
    neptune_token = file.read() 

# Step 1: Initialize Neptune and create new Neptune Run
run = neptune.init(
    project="common/pytorch-integration",
    api_token=neptune_token,
    tags="Neptune Docker"
)

data_dir = "data/CIFAR10"
compressed_ds = "./data/CIFAR10/cifar-10-python.tar.gz"
data_tfms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
}

params = {
    "lr": 1e-2,
    "batch_size": 128,
    "input_size": 32 * 32 * 3,
    "n_classes": 10,
    "model_filename": "basemodel",
}


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

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True)
dataset_size = {"train": len(trainset)}

model = BaseModel(params["input_size"], params["input_size"], params["n_classes"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"])

# Step 2: Log config & pararameters
run["config/dataset/path"] = data_dir
run["config/dataset/transforms"] = data_tfms
run["config/dataset/size"] = dataset_size
run["config/params"] = params


# Step 3: Log losses & metrics
for i, (x, y) in enumerate(trainloader, 0):

    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    # Log batch loss
    run["metrics/training/batch/loss"].log(loss)

    # Log batch accuracy
    run["metrics/training/batch/acc"].log(acc)

    loss.backward()
    optimizer.step()

# Stop logging
run.stop()
