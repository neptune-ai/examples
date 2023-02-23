import neptune
import torch
from fastai.callback.all import SaveModelCallback
from fastai.vision.all import (
    ImageDataLoaders,
    URLs,
    accuracy,
    resnet18,
    untar_data,
    vision_learner,
)
from neptune.integrations.fastai import NeptuneCallback
from neptune.types import File

run = neptune.init_run(
    project="common/fastai-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags="more options",
)

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path, num_workers=0)

# Single & Multi phase logging

# 1. (Neptune) Log a single training phase
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fit_one_cycle(1, cbs=[NeptuneCallback(run=run, base_namespace="experiment_1")])
learn.fit_one_cycle(2)

# 2. (Neptune) Log all training phases of the learner
learn = vision_learner(dls, resnet18, cbs=[NeptuneCallback(run=run, base_namespace="experiment_2")])
learn.fit_one_cycle(1)

# Log model weights

# Add SaveModelCallback
""" You can log your model weight files
  during single training or all training phases
  add  SavemodelCallback() to the callbacks' list
  of your learner or fit method."""

# 1.(Neptune) Log Every N epochs
n = 2
learn = vision_learner(
    dls,
    resnet18,
    metrics=accuracy,
    cbs=[
        SaveModelCallback(every_epoch=n),
        NeptuneCallback(run=run, base_namespace="experiment_3", upload_saved_models="all"),
    ],
)

learn.fit_one_cycle(5)

# 2. (Neptune) Best Model
learn = vision_learner(
    dls,
    resnet18,
    metrics=accuracy,
    cbs=[SaveModelCallback(), NeptuneCallback(run=run, base_namespace="experiment_4")],
)
learn.fit_one_cycle(5)

# 3. (Neptune) Pickling and logging the learner
""" Remove the NeptuneCallback class before pickling the learner object
    to avoid errors due to pickle's inability to pickle local objects
    (i.e., nested functions or methods)"""

pickled_learner = "learner.pkl"
base_namespace = "experiment_5"
neptune_cbk = NeptuneCallback(run=run, base_namespace=base_namespace)
learn = vision_learner(
    dls,
    resnet18,
    metrics=accuracy,
    cbs=[neptune_cbk],
)
learn.fit_one_cycle(1)  # training
learn.remove_cb(neptune_cbk)  # remove NeptuneCallback
learn.export(f"./{pickled_learner}")  # export learner
run[f"{base_namespace}/pickled_learner"].upload(pickled_learner)  # (Neptune) upload pickled learner
learn.add_cb(neptune_cbk)  # add NeptuneCallback back again
learn.fit_one_cycle(1)  # continue training


# (Neptune) Log images
batch = dls.one_batch()
for i, (x, y) in enumerate(dls.decode_batch(batch)):
    # Neptune supports torch tensors
    # fastai uses their own tensor type name TensorImage
    # so you have to convert it back to torch.Tensor
    run["images/one_batch"].append(
        File.as_image(x.as_subclass(torch.Tensor).permute(2, 1, 0) / 255.0),
        name=f"{i}",
        description=f"Label: {y}",
    )
