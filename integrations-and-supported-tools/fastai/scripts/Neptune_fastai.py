import neptune.new as neptune
from fastai.vision.all import (
    resnet18,
    vision_learner,
    ImageDataLoaders,
    untar_data,
    URLs,
)
from fastai.callback.all import SaveModelCallback
from neptune.new.integrations.fastai import NeptuneCallback

run = neptune.init(
    project="common/fastai-integration",
    api_token="ANONYMOUS",
    tags="basic",
)

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

# Log all training phases of the learner
learn = vision_learner(
    dls,
    resnet18,
    cbs=[SaveModelCallback(), NeptuneCallback(run=run, base_namespace="experiment")],
)
learn.fit_one_cycle(2)
learn.fit_one_cycle(1)
