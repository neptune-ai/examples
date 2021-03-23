# Use Neptune API to log your first run

# Step 1: Initialize Neptune and create new run

import neptune.new as neptune

run = neptune.init(project='common/quickstarts',
                   api_token='ANONYMOUS')

# Step 2 - Log metrics during training

import numpy as np
from time import sleep

# log score
run['single_metric'] = 0.62

for i in range(100):
    sleep(0.2) # to see logging live
    run['random_training_metric'].log(i * np.random.random())
    run['other_random_training_metric'].log(0.5 * i * np.random.random())