# Quickstart

# Install ```neptune-client```

get_ipython().system(' pip install --quiet neptune-client==0.5.3')

get_ipython().system(' pip install --upgrade --quiet neptune-client')

# Initialize Neptune

## (option 1) Initialize a public project

import neptune.new as neptune

run = neptune.init(project='common/showroom', api_token='ANONYMOUS')

# Step 4: Log metadata during training

params = {'learning_rate': 0.1}

# log params
run['parameters'] = params

# log name and append tags
run["sys/name"] = 'colab-example'
run["sys/tags"].add(['colab', 'simple'])

# log loss during training
for epoch in range(132):
    run["train/loss"].log(0.97 ** epoch)
    run["train/loss-pow-2"].log((0.97 ** epoch)**2)

# log train and validation scores
run['train/accuracy'] = 0.95
run['valid/accuracy'] = 0.93

# tests
run.wait()

# check losses
assert run['train/loss'].get_last() < 0.97, 'Wrong loss values logged.'
assert run['train/loss-pow-2'].get_last() < 0.97, 'Wrong loss values logged.'

# check tags
all_tags = ['colab', 'simple']
assert set(run["sys/tags"].get()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, run["sys/tags"].get())

# check params
assert run['parameters/learning_rate'].get() == params['learning_rate'], 'Expected: {}, Actual: {}'.format(params['learning_rate'], run['parameters/learning_rate'].get())