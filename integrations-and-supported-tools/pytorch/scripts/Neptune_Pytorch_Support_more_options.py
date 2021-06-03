from numpy.random import permutation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import neptune.new as neptune
import numpy as np

from helpers import *


# Step 1: Initialize Neptune and create new Neptune Run
run = neptune.init(project="common/pytorch-integration", tags='More options script', api_token="ANONYMOUS", source_files=["*.py"])


# Experiment Config
data_dir = "data/CIFAR10"
data_tfms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ])
}

params = {
    "lr": 1e-2,
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "epochs": 1,
    "model_filename": "basemodel",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

# Model & Dataset
class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, n_classes)
        )

    def forward(self, input):
        x = input.view(-1, 32* 32 * 3)
        return self.main(x)

trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], 
                            download=True)
trainloader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=params["bs"],
                                        shuffle=True, num_workers=2)

validset = datasets.CIFAR10(data_dir, train=False,
                        transform=data_tfms["train"],
                        download=True)
validloader = torch.utils.data.DataLoader(validset, 
                                        batch_size=params["bs"], 
                                        num_workers=2)
dataset_size = {"train": len(trainset), "val": len(validset)}

# Instatiate model, criterion and optimizer
model = BaseModel(params["input_sz"], params["input_sz"], params["n_classes"]).to(params["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"])


# Step 2: Log config & hyperpararameters
run["config/dataset/path"] = data_dir
run["config/dataset/transforms"] = data_tfms
run["config/dataset/size"] = dataset_size
run["config/model"] = get_obj_name(model)
run["config/criterion"] = get_obj_name(criterion)
run["config/optimizer"] = get_obj_name(optimizer)
run["config/hyperparameters"] = params

epoch_loss = 0.0
epoch_acc = 0.0
best_acc = 0.0

# Step 3: Log losses and metrics 
for epoch in range(params["epochs"]):
    running_loss = 0.0
    running_corrects = 0
    
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(params["device"]), y.to(params["device"])
        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log batch loss
        run["training/batch/loss"].log(loss)

        # Log batch accuracy
        run["training/batch/acc"].log(acc)

        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        running_corrects += torch.sum(preds == y.data)
        
    epoch_loss = running_loss/dataset_size["train"]
    epoch_acc = running_corrects.double().item() / dataset_size["train"]

    # Log epoch loss
    run[f"training/epoch/loss"].log(epoch_loss)

    # Log epoch accuracy
    run[f"training/epoch/acc"].log(epoch_acc)

    print(f"Epoch:{epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}")
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        
        # Saving model arch & weights
        save_model(model, params["model_filename"])
        print("Saving model -- Done!")


# More options
# Step 4: Log model arch & weights -- > link to adding artifacts
run[f"io_files/artifacts/{params['model_filename']}_arch"].upload(f"./{params['model_filename']}_arch.txt")
run[f"io_files/artifacts/{params['model_filename']}"].upload(f"./{params['model_filename']}.pth")

# Step 5: Log Torch Tensors as images with predictions
save_image_predictions(model, validloader, run["images/predictions"])

# Stop logging
run.stop()

print("Step 6: Explore results in Neptune UI")

