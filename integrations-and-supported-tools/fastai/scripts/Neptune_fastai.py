import fastai
from neptune_fastai.impl import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune

run = neptune.init(project='common/fastai-integration', tags= 'basic', api_token='ANONYMOUS')

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

neptune_cbk = NeptuneCallback(run, 'experiment')
learn = cnn_learner(dls, resnet18, cbs=[neptune_cbk])
learn.fit_one_cycle(1)

run.stop()

