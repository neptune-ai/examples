import fastai
from  neptune.new.integrations.fastai import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune
from neptune.new.types import File

run = neptune.init(project='common/fastai-integration', tags= 'more options', api_token='ANONYMOUS')

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

# Single & Multi phase logging

# 1. Log a single training phase
learn = cnn_learner(dls, resnet18)
learn.fit_one_cycle(1, cbs=[NeptuneCallback(run, 'experiment')])
learn.fit_one_cycle(2)

# 2. Log all training phases of the learner
learn = cnn_learner(dls, resnet18, cbs=[NeptuneCallback(run, 'experiment')])
learn.fit_one_cycle(1)


# Log model weights

# 1. By default NeptuneCallback() saves and logs the best model for you automatically. 
# You can disable it by setting `save_best_model` arg to False.

# 2. Log Every N epochs
n = 1
learn = cnn_learner(dls, resnet18, cbs=[NeptuneCallback(run, 'experiment', save_model_freq=n)])
learn.fit_one_cycle(1)

# 3. Add SaveModel Callback
learn = cnn_learner(dls, resnet18, cbs=[SaveModelCallback(), NeptuneCallback(run, 'experiment')])
learn.fit_one_cycle(1)

# Log images
batch = dls.one_batch()
for i, (x,y) in enumerate(dls.decode_batch(batch)):
    run['images/one_batch'].log(File.as_image(x), name = f'{y}')

run.stop()

