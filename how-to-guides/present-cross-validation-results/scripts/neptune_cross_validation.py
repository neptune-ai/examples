# How to present CV with Neptune

import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from sklearn.model_selection import KFold
from statistics import mean

# Step 1: Create a Neptune Run
run = neptune.init(
    project="common/showroom", tags="cross-validation", api_token="ANONYMOUS"
)

# Step 2: Log config and hyperparameters

# Log Hyperparameters
parameters = {
    "epochs": 2,
    "lr": 1e-2,
    "bs": 10,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "k_folds": 5,
    "model_name": "checkpoint.pth",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}

run["global/params"] = parameters

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

torch.manual_seed(parameters['seed'])
model = BaseModel(
    parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]
).to(parameters["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters["lr"])

# Log model, criterion and optimizer name
run["global/config/model"] = type(model).__name__
run["global/config/criterion"] = type(criterion).__name__
run["global/config/optimizer"] = type(optimizer).__name__

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

validset = datasets.CIFAR10(
    data_dir, train=False, transform=data_tfms["train"], download=True
)

dataset_size = {"train": len(trainset), "val": len(validset)}

dataset = ConcatDataset([trainset, validset])

## Log dataset details
run["global/dataset/CIFAR-10"].track_files(data_dir)
run["global/dataset/dataset_transforms"] = data_tfms
run["global/dataset/dataset_size"] = dataset_size


# Step 3: Log losses and metrics 
def train_step(run, model,trainloader,loss_fn,optimizer,train=True):
    epoch_loss,epoch_acc=0.0,0
    if train: 
        model.train() 
    else:
        model.eval()

    for x, y in trainloader:
        x, y = x.to(parameters["device"]), y.to(parameters["device"])
        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        if train:
            # log batch loss and acc
            run["training/batch/loss"].log(loss)
            run["training/batch/acc"].log(acc)

            loss.backward()
            optimizer.step()
        else: 
            # log batch loss and acc
            run["validation/batch/loss"].log(loss)
            run["validation/batch/acc"].log(acc)

        epoch_acc += torch.sum(preds == y.data).item() 
        epoch_loss += loss.item() * x.size(0)

    epoch_acc = (epoch_acc / len(train_loader.sampler)) * 100
    epoch_loss = epoch_loss / len(train_loader.sampler)

    return epoch_acc, epoch_loss

splits = KFold(n_splits=parameters['k_folds'], shuffle=True)

# K-fold training loop
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=parameters['bs'], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=parameters['bs'], sampler=test_sampler)

    history = {
        'train': 
        {
            'mean_loss': [], 
            'mean_acc': []
        }, 
        'val': 
        {
            'mean_loss': [],
            'mean_acc':[]
        }
    }

    for epoch in range(parameters['epochs']):
        train_acc, train_loss = train_step(run[f'fold_{fold}'],model,train_loader,criterion,optimizer)
        val_acc, val_loss = train_step(run[f'fold_{fold}'],model,test_loader,criterion,optimizer,train=False)

        history['train']['mean_loss'].append(train_loss)
        history['train']['mean_acc'].append(train_acc)
        history['val']['mean_loss'].append(val_loss)
        history['val']['mean_acc'].append(val_acc)

        # log model weights
        torch.save(model.state_dict(), "./" + parameters["model_name"])
        run[f'fold_{fold}/checkpoint'].upload(parameters['model_name'])

history['train']['mean_loss'] = mean(history['train']['loss'])
history['train']['mean_acc'] = mean(history['train']['acc'])
history['val']['mean_loss'] = mean(history['val']['loss'])
history['val']['mean_acc'] = mean(history['val']['acc'])

# log global acc and loss
run['global/metrics'] = history