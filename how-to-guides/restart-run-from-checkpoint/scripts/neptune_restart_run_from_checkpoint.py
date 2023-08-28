import neptune
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neptune.utils import stringify_unsupported
from torch import load as torch_load
from torch import save as torch_save


# (Neptune) Fetch and load checkpoints
def load_checkpoint(run: neptune.Run, epoch: int):
    checkpoint_name = f"epoch_{epoch}"
    ext = run["checkpoints"][checkpoint_name].fetch_extension()
    run["checkpoints"][checkpoint_name].download()  # Download the checkpoint
    checkpoint = torch_load(f"{checkpoint_name}.{ext}")  # Load the checkpoint
    return checkpoint


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


# (Neptune) Initialize existing run
run = neptune.init_run(
    project="common/showroom",  # Replace with your own
    # with_id="SAN-111", # Replace with your run id
)

# (Neptune) Fetch hyperparameters
parameters = run["parameters"].fetch()
parameters["num_epochs"] = 2
run["parameters"] = stringify_unsupported(parameters)

# (Neptune) Fetch and load checkpoint
checkpoints = run.get_structure()["checkpoints"]
epochs = [
    int(checkpoint.split("_")[-1]) for checkpoint in checkpoints
]  # Fetch the epochs of the checkpoints
epochs.sort()  # Sort the epochs
epoch = epochs[-1]  # Fetch the last epoch
checkpoint = load_checkpoint(run, epoch)  # Load the checkpoint


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

# Load model and optimizer state
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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

# Resume training and tracking from checkpoint
for epoch in range(checkpoint["epoch"], parameters["num_epochs"]):
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
