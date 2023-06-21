import neptune
import numpy as np
import torch
import tqdm
from neptune.types import File
from neptune.utils import stringify_unsupported
from neptune_tensorboard import enable_tensorboard_logging
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/quickstarts",  # replace with your own
)

enable_tensorboard_logging(run)

writer = SummaryWriter()

# Hyperparams for training
parameters = {
    "lr": 1e-2,
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "model_filename": "basemodel",
    "epochs": 2,
}

writer.add_hparams(parameters, {})

parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

writer.add_text("model_summary", str(model))

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

# Log metrics while training
for epoch in tqdm.tqdm(range(parameters["epochs"])):
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(parameters["device"]), y.to(parameters["device"])
        optimizer.zero_grad()
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log after every 30 steps
        if i % 30 == 0:
            writer.add_scalar("train_loss", loss.item())
            writer.add_scalar("train_acc", acc.item())

        loss.backward()
        optimizer.step()

# Log prediction from model
dataiter = iter(validloader)
images, labels = next(dataiter)

# Predict batch of n_samples
n_samples = 10
imgs = images[:n_samples].to(parameters["device"])
probs = torch.nn.functional.softmax(model(imgs), dim=1)

# Decode probs and log tensors as image
for i, ps in enumerate(probs):
    pred = classes[torch.argmax(ps)]
    ground_truth = classes[labels[i]]
    description = f"pred: {pred} | ground truth: {ground_truth}"

    img = imgs[i].cpu().squeeze().permute(2, 1, 0).clip(0, 1)
    name = f"{i}_{pred}_{ground_truth}"
    # Log Series of Tensors as Image and Predictions.
    writer.add_image(name, img, dataformats="HWC")
