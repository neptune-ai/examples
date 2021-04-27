import numpy as np
from time import sleep
import neptune.new as neptune

run = neptune.init(project='common/quickstarts', api_token='ANONYMOUS')

# log score
run['single_metric'] = 0.62

for i in range(100):
    sleep(0.2) # to see logging live
    run['random_training_metric'].log(i * np.random.random())
    run['other_random_training_metric'].log(0.5 * i * np.random.random())
