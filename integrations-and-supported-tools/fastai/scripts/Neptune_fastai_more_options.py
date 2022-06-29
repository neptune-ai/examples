import neptune.new as neptune
import torch
from fastai.callback.all import SaveModelCallback
from fastai.vision.all import (
    resnet18,
    vision_learner,
    ImageDataLoaders,
    untar_data,
    URLs,
    accuracy,
)
from neptune.new.integrations.fastai import NeptuneCallback
from neptune.new.types import File

run = neptune.init(
    project="common/fastai-integration",
    api_token="ANONYMOUS",
    tags="more options",
)

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

# Single & Multi phase logging

# 1. Log a single training phase
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fit_one_cycle(1, cbs=[NeptuneCallback(run=run, base_namespace="experiment_1")])
learn.fit_one_cycle(2)

# 2. Log all training phases of the learner
learn = vision_learner(
    dls, resnet18, cbs=[NeptuneCallback(run=run, base_namespace="experiment_2")]
)
learn.fit_one_cycle(1)

# Log model weights

# Add SaveModelCallback
""" You can log your model weight files
  during single training or all training phases 
  add  SavemodelCallback() to the callbacks' list 
  of your learner or fit method."""

# 1. Log Every N epochs
n = 2
learn = vision_learner(
    dls,
    resnet18,
    metrics=accuracy,
    cbs=[
        SaveModelCallback(every_epoch=n),
        NeptuneCallback(
            run=run, base_namespace="experiment_3", upload_saved_models="all"
        ),
    ],
)

learn.fit_one_cycle(5)

# 2. Best Model
learn = vision_learner(
    dls,
    resnet18,
    metrics=accuracy,
    cbs=[SaveModelCallback(), NeptuneCallback(run=run, base_namespace="experiment_4")],
)
learn.fit_one_cycle(5)

# Log images
batch = dls.one_batch()
for i, (x, y) in enumerate(dls.decode_batch(batch)):
    # Neptune supports torch tensors
    # fastai uses their own tensor type name TensorImage
    # so you have to convert it back to torch.Tensor
    run["images/one_batch"].log(
        File.as_image(x.as_subclass(torch.Tensor).permute(2, 1, 0) / 255.0),
        name=f"{i}",
        description=f"Label: {y}",
    )
