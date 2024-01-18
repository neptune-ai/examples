# Neptune + MosaicML Composer

## Import dependencies
import torch
from composer import Trainer
from composer.algorithms import LabelSmoothing, ProgressiveResizing
from composer.callbacks import ImageVisualizer
from composer.loggers import NeptuneLogger
from composer.models import mnist_model
from neptune import ANONYMOUS_API_TOKEN
from neptune.types import File
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

## Prepare dataset and dataloaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST("data", download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST("data", download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32)
eval_dataloader = DataLoader(eval_dataset, batch_size=16)

## (Neptune) Create `neptune_logger`
neptune_logger = NeptuneLogger(
    api_token=ANONYMOUS_API_TOKEN,  # Replace with your own
    project="common/mosaicml-composer",  # Replace with your own
    tags=["mnist", "script"],  # (optional) use your own
)

## Configure Composer algorithms
label_smoothing = LabelSmoothing(0.1)

prog_resize = ProgressiveResizing(
    initial_scale=0.6,
    finetune_fraction=0.34,
)

## Train model
trainer = Trainer(
    model=mnist_model(),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="3ep",
    device="gpu" if torch.cuda.is_available() else "cpu",
    algorithms=[label_smoothing, prog_resize],
    callbacks=ImageVisualizer(),  # Adding the ImageVisualizer() callback automatically logs input images to Neptune
    loggers=neptune_logger,
)

trainer.fit()

## Log additional metadata

neptune_logger.base_handler["sample_image"].upload(File.as_image(train_dataset.data[0] / 255))

## Log metadata to your custom namespace
neptune_logger.neptune_run["eval/sample_image"].upload(File.as_image(eval_dataset.data[0] / 255))

## Stop logging
trainer.close()

## Analyze run in the Neptune app
# Follow the run link in the console output and explore the logged metadata.

# You can also explore this example run
# https://app.neptune.ai/showcase/mosaicml-composer/e/MMLCOMP-3/metadata
