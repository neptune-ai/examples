from time import sleep

import neptune.new as neptune
import numpy as np

run = neptune.init_run(project="common/quickstarts", api_token=neptune.ANONYMOUS_API_TOKEN)

# log score
run["single_metric"] = 0.62

for i in range(100):
    sleep(0.2)  # to see logging live
    run["random_training_metric"].append(i * np.random.random())
    run["other_random_training_metric"].append(0.5 * i * np.random.random())
