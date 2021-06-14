import fastai
from  neptune.new.integrations.fastai import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune

run = neptune.init(
    project='common/fastai-integration', 
    api_token='ANONYMOUS', 
    tags= 'basic'
)

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

# Log all training phases of the learner
learn = cnn_learner(dls, resnet18, cbs=[NeptuneCallback(run, 'experiment')])
learn.fit_one_cycle(2)
learn.fit_one_cycle(1)

run.stop()

