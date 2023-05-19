from functools import reduce
from statistics import mean

import neptune
import torch
import torch.nn as nn
from neptune.utils import stringify_unsupported
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange

# Create a Neptune Run
run = neptune.init_run(
    project="common/showroom",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags="cross-validation",
)


# Log config and hyperparameters
parameters = {
    "epochs": 2,
    "learning_rate": 1e-2,
    "batch_size": 10,
    "image_size": (3, 32, 32),
    "n_classes": 10,
    "k_folds": 2,
    "checkpoint_name": "checkpoint.pth",
    "dataset_size": 1000,
    "seed": 42,
}

image_size = reduce(lambda x, y: x * y, parameters["image_size"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Log hyperparameters
run["parameters"] = stringify_unsupported(parameters)
run["parameters/device"] = str(device)

# Seed
torch.manual_seed(parameters["seed"])


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
        x = input.view(-1, image_size)
        return self.main(x)


torch.manual_seed(parameters["seed"])
model = BaseModel(
    image_size,
    image_size,
    parameters["n_classes"],
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=parameters["learning_rate"])

# Log model, criterion and optimizer name
run["parameters/model/name"] = type(model).__name__
run["parameters/model/criterion"] = type(criterion).__name__
run["parameters/model/optimizer"] = type(optimizer).__name__

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
dataset = datasets.FakeData(
    size=parameters["dataset_size"],
    image_size=parameters["image_size"],
    num_classes=parameters["n_classes"],
    transform=data_tfms["train"],
)

# Log dataset details
run["dataset/transforms"] = stringify_unsupported(data_tfms)
run["dataset/size"] = parameters["dataset_size"]

splits = KFold(n_splits=parameters["k_folds"], shuffle=True)
epoch_acc_list, epoch_loss_list = [], []

# Log losses and metrics per fold
for fold, (train_ids, _) in tqdm(enumerate(splits.split(dataset))):
    train_sampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(dataset, batch_size=parameters["batch_size"], sampler=train_sampler)
    for _ in trange(parameters["epochs"]):
        epoch_acc, epoch_loss = 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model.forward(x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)
            acc = (torch.sum(preds == y.data)) / len(x)

            # Log batch loss and acc
            run[f"fold_{fold}/training/batch/loss"].append(loss)
            run[f"fold_{fold}/training/batch/acc"].append(acc)

            loss.backward()
            optimizer.step()

        epoch_acc += torch.sum(preds == y.data).item()
        epoch_loss += loss.item() * x.size(0)

    epoch_acc_list.append((epoch_acc / len(train_loader.sampler)) * 100)
    epoch_loss_list.append(epoch_loss / len(train_loader.sampler))

    # Log model checkpoint
    torch.save(model.state_dict(), f"./{parameters['checkpoint_name']}")
    run[f"fold_{fold}/checkpoint"].upload(parameters["checkpoint_name"])
    run.sync()

# Log mean of metrics across all folds
run["results/metrics/train/mean_acc"] = mean(epoch_acc_list)
run["results/metrics/train/mean_loss"] = mean(epoch_loss_list)
