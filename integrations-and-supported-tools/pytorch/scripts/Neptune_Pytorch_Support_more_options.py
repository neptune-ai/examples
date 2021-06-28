from numpy.random import permutation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import neptune.new as neptune
import numpy as np
from neptune.new.types import File
import torch.nn.functional as F


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
                                        shuffle=True)

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

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog","horse","ship","truck"]


# Step 2: Log config & hyperpararameters
run["config/dataset/path"] = data_dir
run["config/dataset/transforms"] = data_tfms
run["config/dataset/size"] = dataset_size
run["config/model"] = type(model).__name__
run["config/criterion"] = type(criterion).__name__
run["config/optimizer"] = type(optimizer).__name__
run["config/hyperparameters"] = params
run["config/classes"] = classes

# Step 3: Log losses and metrics 
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


## More options

# Step 4: Saving model
fname = params["model_filename"]

# Saving model architecture to .txt
with open(f"./{fname}_arch.txt", "w") as f:  f.write(str(model))
# Saving model weights .pth
torch.save(model.state_dict(), f"./{fname}.pth")

# Step 4.1: Log model archictecture & weights  
run[f"io_files/artifacts/{params['model_filename']}_arch"].upload(f"./{params['model_filename']}_arch.txt")
run[f"io_files/artifacts/{params['model_filename']}"].upload(f"./{params['model_filename']}.pth")

# Step 5: Log Torch Tensors as images with predictions

# Getting batch
dataiter = iter(validloader)
images, labels = dataiter.next()
model.eval()

# Moving model to cpu for inference 
if torch.cuda.is_available(): model.to("cpu")

# Predict batch of n_samples
n_samples = 50
imgs = images[:n_samples]
probs = F.softmax(model(imgs),dim=1)


# Decode probs and Log tensors as image
for i, ps in enumerate(probs):
    pred = classes[torch.argmax(ps)]
    ground_truth = classes[labels[i]]
    description = "\n".join(
        ["class {}: {}%".format(classes[n], np.round(p.detach().numpy() * 100, 2)) for n, p in enumerate(ps)]
    )
    # Log Series of Tensors as Image and Predictions. 
    run["images/predictions"].log(
        File.as_image(imgs[i].squeeze().permute(2,1,0).clip(0,1)), 
        name=f'{i}_{pred}_{ground_truth}', 
        description=description
    )

# Stop logging
run.stop()

print("Step 6: Explore results in Neptune UI")

