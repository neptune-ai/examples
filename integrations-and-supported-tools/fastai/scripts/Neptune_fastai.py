
import fastai

fastai.__version__

from neptune_fastai.impl import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune

"""run"""

run = neptune.init(project='common/fastai-integration', api_token='ANONYMOUS')

"""data"""

path = untar_data(URLs.MNIST_TINY)

path.ls()

dls = ImageDataLoaders.from_csv(path)

dls.show_batch()

learn = cnn_learner(dls, resnet18, cbs=[NeptuneCallback(run, 'experiment')])

learn.fit_one_cycle(1)

learn.fine_tune(2)

??NeptuneCallback()

run.stop()

