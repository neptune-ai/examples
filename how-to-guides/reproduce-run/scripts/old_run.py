import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from neptune.utils import stringify_unsupported
from torchvision import datasets, transforms

###################################################################
# (Neptune) Step 1: Initialize Neptune and create new Neptune run #
###################################################################
run = neptune.init_run(
    project="common/showroom",
    tags="Basic script",
    api_token=neptune.ANONYMOUS_API_TOKEN,
)

# Experiment Config
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
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "model_filename": "basemodel",
}


# Model & Dataset
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


trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=params["bs"], shuffle=True, num_workers=0
)
dataset_size = {"train": len(trainset)}

# Instatiate model, criterion and optimizer
model = BaseModel(params["input_sz"], params["input_sz"], params["n_classes"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"])

###############################################
# (Neptune) Step 2: Log config & pararameters #
###############################################
run["config/dataset/path"] = data_dir
run["config/dataset/transforms"] = stringify_unsupported(data_tfms)
run["config/dataset/size"] = dataset_size
run["config/params"] = stringify_unsupported(params)

##########################################
# (Neptune) Step 3: Log losses & metrics #
##########################################
for i, (x, y) in enumerate(trainloader, 0):
    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    run["training/batch/loss"].append(loss)

    run["training/batch/acc"].append(acc)

    loss.backward()
    optimizer.step()
