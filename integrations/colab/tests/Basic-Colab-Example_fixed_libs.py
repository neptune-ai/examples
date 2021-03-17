# Quickstart

# Install ```neptune-client```

get_ipython().system(' pip install --quiet git+https://github.com/neptune-ai/neptune-client.git@alpha')

# Initialize Neptune

## (option 1) Initialize a public project

import neptune.new as neptune

exp = neptune.init(project='common/showroom', api_token='ANONYMOUS')

# Step 4: Log metadata during training

params = {'learning_rate': 0.1}

# log params
exp['parameters'] = params

# log name and append tags
exp["sys/name"] = 'colab-example'
exp["sys/tags"].add(['colab', 'simple'])

# log loss during training
for epoch in range(132):
    exp["train/loss"].log(0.97 ** epoch)
    exp["train/loss-pow-2"].log((0.97 ** epoch)**2)

# log train and validation scores
exp['train/accuracy'] = 0.95
exp['valid/accuracy'] = 0.93

# tests
exp.wait()

# check losses
assert exp['train/loss'].get_last() < 0.97, 'Wrong loss values logged.'
assert exp['train/loss-pow-2'].get_last() < 0.97, 'Wrong loss values logged.'

# check tags
all_tags = ['colab', 'simple']
assert set(exp["sys/tags"].get()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, exp["sys/tags"].get())

# check params
learning_rate: 0.1

assert exp['parameters/learning_rate'].get() == learning_rate, 'Expected: {}, Actual: {}'.format(learning_rate, exp['parameters/learning_rate'].get())