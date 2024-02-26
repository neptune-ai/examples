# Neptune + MosaicML Composer

# Script adapted from https://docs.mosaicml.com/projects/composer/en/latest/examples/getting_started.html
# Date accessed: 2024-02-06

import composer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from composer.algorithms import LabelSmoothing, ProgressiveResizing
from composer.callbacks import ImageVisualizer
from composer.loggers import NeptuneLogger
from composer.models import ComposerClassifier
from neptune.types import File
from torchvision import datasets, transforms

data_directory = "./data"
batch_size = 512

transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(data_directory, train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(data_directory, train=False, download=True, transform=transforms)

## Set-up data-loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


## Create model
class Model(nn.Module):
    """Toy convolutional neural network architecture in pytorch for MNIST."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (4, 4))
        out = torch.flatten(out, 1, -1)
        out = self.fc1(out)
        out = F.relu(out)
        return self.fc2(out)


model = ComposerClassifier(module=Model(num_classes=10))

## Use composer algorithms
label_smoothing = LabelSmoothing(
    0.1
)  # We're creating an instance of the LabelSmoothing algorithm class

prog_resize = ProgressiveResizing(
    initial_scale=0.6,  # Size of images at the beginning of training = .6 * default image size
    finetune_fraction=0.34,  # Train on default size images for 0.34 of total training time.
)

algorithms = [label_smoothing, prog_resize]

## (Neptune) Create `neptune_logger`
from neptune import ANONYMOUS_API_TOKEN

neptune_logger = NeptuneLogger(
    api_token=ANONYMOUS_API_TOKEN,  # or replace with your own
    project="common/mosaicml-composer",  # or replace with your own
    tags=["mnist", "script"],  # (optional) use your own
)

## Train model
train_epochs = "3ep"  # Train for 3 epochs because we're assuming Colab environment and hardware
device = "gpu" if torch.cuda.is_available() else "cpu"  # select the device

trainer = composer.trainer.Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    max_duration=train_epochs,
    device=device,
    callbacks=ImageVisualizer(),  # Adding the ImageVisualizer() callback automatically logs input images to Neptune
    loggers=neptune_logger,
    algorithms=algorithms,
)

trainer.fit()

## (Neptune) Log additional metadata

neptune_logger.base_handler["sample_image"].upload(File.as_image(train_dataset.data[0] / 255))

## Log metadata to your custom namespace
neptune_logger.neptune_run["eval/sample_image"].upload(File.as_image(test_dataset.data[0] / 255))

## Stop logging
trainer.close()

## Analyze run in the Neptune app
# To explore the logged metadata, follow the run link in the console output.

# You can also explore this example run:
# https://app.neptune.ai/showcase/mosaicml-composer/e/MMLCOMP-3/metadata
