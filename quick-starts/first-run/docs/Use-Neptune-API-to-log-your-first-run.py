# Use Neptune API to Iog your first experiment

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