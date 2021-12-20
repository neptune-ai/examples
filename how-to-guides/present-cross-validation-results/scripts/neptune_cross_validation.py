# How to present CV with Neptune
import neptune.new as neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import KFold
from statistics import mean

# Step 1: Create a Neptune Run
run = neptune.init(
    project="common/showroom", tags="cross-validation", api_token="ANONYMOUS"
)

# Step 2: Log config and hyperparameters
parameters = {
    "epochs": 1,
    "lr": 1e-2,
    "bs": 10,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "k_folds": 2,
    "model_name": "checkpoint.pth",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}

# Log hyperparameters
run["global/parameters"] = parameters

# Seed
torch.manual_seed(parameters['seed'])

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
    parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]
).to(parameters["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters["lr"])

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

run["global/dataset/CIFAR-10"].track_files(data_dir)
run["global/dataset/dataset_transforms"] = data_tfms
run["global/dataset/dataset_size"] = dataset_size

splits = KFold(n_splits=parameters['k_folds'], shuffle=True)
epoch_acc_list, epoch_loss_list= [], []

for fold, (train_ids, _ ) in enumerate(splits.split(trainset)):
    train_sampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(trainset, batch_size=parameters['bs'], sampler=train_sampler)
    for epoch in range(parameters["epochs"]): 
        epoch_acc, epoch_loss= 0, 0.0
        for x, y in train_loader:
            x, y = x.to(parameters["device"]), y.to(parameters["device"])
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
    torch.save(model.state_dict(), f"./{parameters['model_name']}")
    run[f'fold_{fold}/checkpoint'].upload(parameters['model_name'])
    
run["global/metrics/train/mean_acc"] = mean(epoch_acc_list)
run["global/metrics/train/mean_loss"] = mean(epoch_loss_list)  