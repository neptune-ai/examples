import fastai
from neptune.new.integrations.fastai import NeptuneCallback
from fastai.vision.all import *
import neptune.new as neptune
from neptune.new.types import File

run = neptune.init(
    project='common/fastai-integration',
    api_token='ANONYMOUS', 
    tags= 'more options'
)

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_csv(path)

# Single & Multi phase logging

# 1. Log a single training phase
learn = cnn_learner(dls, resnet18, metrics= accuracy)
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

# 3. Add SaveModelCallback
# If you want to log your model weight files during 
# single training phase then add SavemodelCallback().
learn.fit_one_cycle(1, cbs=[SaveModelCallback(), NeptuneCallback(run, 'experiment')])

# Log images
batch = dls.one_batch()
for i, (x,y) in enumerate(dls.decode_batch(batch)):
    # Neptune supports torch tensors
    # fastai uses their own tensor type name TensorImage 
    # so you have to convert it back to torch.Tensor
    run['images/one_batch'].log(
        File.as_image(x.as_subclass(torch.Tensor).permute(2,1,0)/255.), 
        name = f'{i}', description = f'Label: {y}')
    
# Stop Run
run.stop()