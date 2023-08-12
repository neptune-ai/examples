import neptune
import numpy as np
import torch
from neptune.types import File
from neptune.utils import stringify_unsupported
from neptune_pytorch import NeptuneLogger
from torch import nn, optim
from torchvision import datasets, transforms

run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/pytorch-integration",  # replace with your own
)

# Hyperparams for training
parameters = {
    "lr": 1e-2,
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "model_filename": "basemodel",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "epochs": 2,
}


# Model
class Model(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(Model, self).__init__()
        self.seq_model = nn.Sequential(
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
        return self.seq_model(x)


model = Model(parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]).to(
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
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=parameters["bs"], shuffle=True, num_workers=0
)
validset = datasets.CIFAR10(data_dir, train=False, transform=data_tfms["train"], download=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=parameters["bs"], num_workers=0)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# (Neptune) Create NeptuneLogger
npt_logger = NeptuneLogger(
    run,
    model=model,
    log_model_diagram=True,
    log_gradients=True,
    log_parameters=True,
    log_freq=30,
)

# (Neptune) Log hyperparams
# NOTE: The base_namespace attribute of the logger can be used to log metadata consistently
# under the 'base_namespace' namespace.
run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(parameters)

# (Neptune) Log metrics while training
for epoch in range(parameters["epochs"]):
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(parameters["device"]), y.to(parameters["device"])
        optimizer.zero_grad()
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log after every 30 steps
        if i % 30 == 0:
            run[npt_logger.base_namespace]["batch/loss"].append(loss.item())
            run[npt_logger.base_namespace]["batch/acc"].append(acc.item())

        loss.backward()
        optimizer.step()

    # Checkpoint number is automatically incremented on subsequent call.
    # Call 1 -> ckpt_1.pt
    # Call 2 -> ckpt_2.pt
    # npt_logger.log_checkpoint()  # uncomment to log checkpoint to the run

# (Neptune) Log prediction from model
dataiter = iter(validloader)
images, labels = next(dataiter)

# Predict batch of n_samples
n_samples = 10
imgs = images[:n_samples].to(parameters["device"])
probs = torch.nn.functional.softmax(model(imgs), dim=1)

# Decode probs and Log tensors as image
for i, ps in enumerate(probs):
    pred = classes[torch.argmax(ps)]
    ground_truth = classes[labels[i]]
    description = f"pred: {pred} | ground truth: {ground_truth}"

    # Log Series of Tensors as Image and Predictions.
    run[npt_logger.base_namespace]["predictions"].append(
        File.as_image(imgs[i].cpu().squeeze().permute(2, 1, 0).clip(0, 1)),
        name=f"{i}_{pred}_{ground_truth}",
        description=description,
    )

# (Neptune) Log final model as "model.pt"
# npt_logger.log_model("model")  # uncomment to log final model to the run

# (Neptune) Stop logging
run.stop()
