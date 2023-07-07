import neptune
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neptune.utils import stringify_unsupported
from torch import save as torch_save


# (Neptune) Save and log checkpoints while training
def save_checkpoint(
    run: neptune.Run, model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: torch.tensor
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss.item(),
    }
    checkpoint_name = f"checkpoint-{epoch}-{loss:.2f}.pth"
    torch_save(checkpoint, checkpoint_name)  # Save the checkpoint locally
    run[f"checkpoints/epoch_{epoch}"].upload(checkpoint_name)  # Upload to Neptune


# (Neptune) Initialize a new run
run = neptune.init_run(
    project="common/showroom",  # Replace with your own
)

# Hyperparams for training
parameters = {
    "lr": 1e-2,
    "batch_size": 128,
    "input_size": 32 * 32 * 3,
    "n_classes": 10,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_epochs": 1,
    "ckpt_frequency": 5,
}

# (Neptune) Log hyperparameters
run["parameters"] = stringify_unsupported(parameters)


# Model
class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, n_classes):
        super(Model, self).__init__()
        self.seq_model = nn.Sequential(
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
        return self.seq_model(x)


model = Model(parameters["input_size"], parameters["input_size"], parameters["n_classes"]).to(
    parameters["device"]
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters["lr"])

# Data
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
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=parameters["batch_size"], shuffle=True, num_workers=0
)

for epoch in range(parameters["num_epochs"]):
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(parameters["device"]), y.to(parameters["device"])
        optimizer.zero_grad()
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # (Neptune) Log metrics
        run["metrics"]["batch/loss"].append(loss.item())
        run["metrics"]["batch/acc"].append(acc.item())

        loss.backward()
        optimizer.step()

    if epoch % parameters["ckpt_frequency"] == 0:
        # (Neptune) Log checkpoints
        save_checkpoint(run, model, optimizer, epoch, loss)
