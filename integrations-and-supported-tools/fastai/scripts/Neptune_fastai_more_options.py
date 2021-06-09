import fastai
from neptune_fastai.impl import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune
from neptune.new.types import File

run = neptune.init(project='common/fastai-integration', tags= 'more options', api_token='ANONYMOUS')

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

neptune_cbk = NeptuneCallback(run, 'experiment')
learn = cnn_learner(dls, resnet18, cbs=[neptune_cbk])
learn.fit_one_cycle(1)

batch = dls.one_batch()
for i, (x,y) in enumerate(dls.decode_batch(batch)):
    run['images/one_batch'].log(File.as_image(x), name = f'{y}')

run.stop()

