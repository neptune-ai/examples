# Use Neptune API to Iog your first experiment

# Before you start

get_ipython().system(' pip install --quiet git+https://github.com/neptune-ai/neptune-client.git@alpha')

# Step 1: Initialize Neptune and create new experiment

import neptune.alpha as neptune

exp = neptune.init(project='common/quickstarts',
                   api_token='ANONYMOUS')

# Step 2 - Log metrics during training

import numpy as np
from time import sleep

# log score
exp['single_metric'] = 0.62

for i in range(100):
    sleep(0.2) # to see logging live
    exp['random_training_metric'].log(i * np.random.random())
    exp['other_random_training_metric'].log(0.5 * i * np.random.random())

# tests
exp.wait()

# check score
sm = 0.62

assert exp['single_metric'].get() == sm, 'Expected: {}, Actual: {}'.format(sm, exp['single_metric'].get())

# check metrics
assert isinstance(exp['random_training_metric'].get_last(), float), 'Incorrect metric type'
assert isinstance(exp['other_random_training_metric'].get_last(), float), 'Incorrect metric type'