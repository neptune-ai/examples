from statistics import mean

import neptune.new as neptune
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# Step 1: Create a Neptune Run
run = neptune.init_run(
    project="common/showroom",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags="cross-validation",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step 2: Log config and hyperparameters
parameters = {
    "epochs": 3,
    "learning_rate": 1e-2,
    "batch_size": 10,
    "input_size": 32 * 32 * 3,
    "n_classes": 10,
    "k_folds": 5,
    "checkpoint_name": "checkpoint.pth",
    "seed": 42,
}

# Log hyperparameters
run["parameters"] = parameters

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
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


torch.manual_seed(parameters["seed"])
model = BaseModel(parameters["input_size"], parameters["input_size"], parameters["n_classes"]).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=parameters["learning_rate"])

# Log model, criterion and optimizer name
run["parameters/model/name"] = type(model).__name__
run["parameters/model/criterion"] = type(criterion).__name__
run["parameters/model/optimizer"] = type(optimizer).__name__

# trainset
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
trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
dataset_size = len(trainset)

# Log dataset details
run["dataset/CIFAR-10"].track_files(data_dir)
run["dataset/transforms"] = data_tfms
run["dataset/size"] = dataset_size

splits = KFold(n_splits=parameters["k_folds"], shuffle=True)
epoch_acc_list, epoch_loss_list = [], []

# Step 3: Log losses and metrics per fold
from torch.utils.data import DataLoader, SubsetRandomSampler

for fold, (train_ids, _) in enumerate(splits.split(trainset)):
    train_sampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(trainset, batch_size=parameters["batch_size"], sampler=train_sampler)
    for epoch in range(parameters["epochs"]):
        epoch_acc, epoch_loss = 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model.forward(x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)
            acc = (torch.sum(preds == y.data)) / len(x)

            # Log batch loss and acc
            run[f"fold_{fold}/training/batch/loss"].log(loss)
            run[f"fold_{fold}/training/batch/acc"].log(acc)

            loss.backward()
            optimizer.step()

        epoch_acc += torch.sum(preds == y.data).item()
        epoch_loss += loss.item() * x.size(0)

    epoch_acc_list.append((epoch_acc / len(train_loader.sampler)) * 100)
    epoch_loss_list.append(epoch_loss / len(train_loader.sampler))

    # Log model checkpoint
    torch.save(model.state_dict(), f"./{parameters['checkpoint_name']}")
    run[f"fold_{fold}/checkpoint"].upload(parameters["checkpoint_name"])

# Log mean of metrics across all folds
run["results/metrics/train/mean_acc"] = mean(epoch_acc_list)
run["results/metrics/train/mean_loss"] = mean(epoch_loss_list)
