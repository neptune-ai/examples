import neptune
from fastai.callback.all import SaveModelCallback
from fastai.vision.all import (
    ImageDataLoaders,
    URLs,
    resnet18,
    untar_data,
    vision_learner,
)
from neptune.integrations.fastai import NeptuneCallback

run = neptune.init_run(
    project="common/fastai-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags="basic",
)

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path, num_workers=0)

# Log all training phases of the learner
learn = vision_learner(
    dls,
    resnet18,
    cbs=[SaveModelCallback(), NeptuneCallback(run=run, base_namespace="experiment")],
)
learn.fit_one_cycle(2)
learn.fit_one_cycle(1)
