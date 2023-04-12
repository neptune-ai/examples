import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment
from torchvision import datasets, transforms

if torch.device("cuda:0"):
    torch.cuda.empty_cache()

# Initialize Neptune and create new Neptune run
neptune_run = neptune.init_run(
    project="common/sacred-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags="more_options",
)

# Add NeptuneObserver() to your sacred experiment's observers
ex = Experiment("image_classification")
ex.observers.append(NeptuneObserver(run=neptune_run))


class BaseModel(nn.Module):
    def __init__(self, input_sz=32 * 32 * 3, n_classes=10):
        super(BaseModel, self).__init__()
        self.lin = nn.Linear(input_sz, n_classes)

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.lin(x)


model = BaseModel()


# Log hyperparameters
@ex.config
def cfg():
    data_dir = "data/CIFAR10"
    data_tfms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    }
    lr = 1e-2
    bs = 128
    n_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Log loss and metrics
@ex.main
def run(data_dir, data_tfms, n_classes, lr, bs, device, _run):
    trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log loss
        ex.log_scalar("training/batch/loss", loss)
        # Log accuracy
        ex.log_scalar("training/batch/acc", acc)

        loss.backward()
        optimizer.step()

    return {"final_loss": loss.item(), "final_acc": acc.cpu().item()}


# Run you experiment and explore metadata in the Neptune app
ex.run()

# More Options
# Log Artifacts (Model architecture and weights)

# Save model architecture
model_fname = "model"
print(f"Saving model archictecture as {model_fname}.txt")
with open(f"{model_fname}_arch.txt", "w") as f:
    f.write(str(model))

# Save model weights
print(f"Saving model weights as {model_fname}.pth")
torch.save(model.state_dict(), f"./{model_fname}.pth")

# Log model architecture and weights
ex.add_artifact(filename=f"./{model_fname}_arch.txt", name=f"{model_fname}_arch")

ex.add_artifact(filename=f"./{model_fname}.pth", name=model_fname)
