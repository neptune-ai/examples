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