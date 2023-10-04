# Neptune + MosaicML Composer

## Import dependencies
import torch

from neptune import ANONYMOUS_API_TOKEN
from neptune.types import File

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from composer import Trainer
from composer.models import mnist_model
from composer.loggers import NeptuneLogger
from composer.algorithms import LabelSmoothing, BlurPool, ProgressiveResizing


## Prepare dataset and dataloaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('data', download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST('data', download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)
eval_dataloader = DataLoader(eval_dataset, batch_size=128)

## (Neptune) Create `neptune_logger`
neptune_logger = NeptuneLogger(
    api_token=ANONYMOUS_API_TOKEN,  # Replace with your own
    project="common/mosaicml-composer",  # Replace with your own
    tags=["mnist", "script"],  # (optional) use your own
)

## Configure Composer algorithms
label_smoothing = LabelSmoothing(0.1)

blurpool = BlurPool(
    replace_convs=True,
    replace_maxpools=True,
    blur_first=True,
)

prog_resize = ProgressiveResizing(
    initial_scale=.6,
    finetune_fraction=0.34,
)

## Train model
trainer = Trainer(
    model=mnist_model(),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="3ep",
    device="gpu" if torch.cuda.is_available() else "cpu",
    algorithms = [label_smoothing, blurpool, prog_resize],
    loggers=neptune_logger,
)

trainer.fit()

## Log additional metadata
neptune_logger.base_handler["images"].extend([File.as_image(img/255) for img in train_dataset.data[:50]])

## Log metadata to your custom namespace
neptune_logger.neptune_run["eval/images"].extend([File.as_image(img/255) for img in eval_dataset.data[:50]])

## Stop logging
trainer.close()

## Analyze run in the Neptune app
# Follow the run link in the console output and explore the logged metadata.
# You can also explore this example run
# https://app.neptune.ai/o/common/org/mosaicml/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=Overview-99f571df-0fec-4447-9ffe-5a4c668577cd&shortId=CAT-2
