import fastai
from neptune_fastai.impl import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune

run = neptune.init(project='common/fastai-integration', tags= 'basic', api_token='ANONYMOUS')

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

neptune_cbk = NeptuneCallback(run, 'experiment')
learn = cnn_learner(dls, resnet18)

# Log a single training phase
learn.fit(1, cbs=[neptune_cbk])

# Log all training phases of the learner
learn = cnn_learner(dls, resnet18, cbs=[neptune_cbk])
learn.fit_one_cycle(1)
learn.fit_one_cycle(2)

run.stop()

