# Monitor ML runs live 

# Introduction
# 
# This guide will show you how to:
# 
# * Monitor training and evaluation metrics and losses live
# * Monitor hardware resources during training
# 
# By the end of it, you will monitor your metrics, losses, and hardware live in Neptune!

# Setup

get_ipython().system(' pip install --quiet git+https://github.com/neptune-ai/neptune-client.git@alpha tensorflow==2.3.1')

get_ipython().system(' pip install --upgrade --quiet git+https://github.com/neptune-ai/neptune-client.git@alpha tensorflow')

# Step 1: Create a basic training script

from tensorflow import keras

# parameters
PARAMS = {'epoch_nr': 10,
          'batch_size': 256,
          'lr': 0.005,
          'momentum': 0.4,
          'use_nesterov': True,
          'unit_nr': 256,
          'dropout': 0.05}

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(PARAMS['unit_nr'], activation=keras.activations.relu),
    keras.layers.Dropout(PARAMS['dropout']),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])

optimizer = keras.optimizers.SGD(lr=PARAMS['lr'],
                                 momentum=PARAMS['momentum'],
                                 nesterov=PARAMS['use_nesterov'], )

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 2: Initialize Neptune and create new experiment

import neptune.alpha as neptune

exp = neptune.init(project='common/quickstarts',
                   api_token='ANONYMOUS')

# Step 3: Add logging for metrics and losses

class NeptuneLogger(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            exp['batch/{}'.format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            exp['epoch/{}'.format(log_name)].log(log_value)

model.fit(x_train, y_train,
          epochs=PARAMS['epoch_nr'],
          batch_size=PARAMS['batch_size'],
          callbacks=[NeptuneLogger()])

# tests
exp.wait()

# check metrics
assert isinstance(exp['epoch/accuracy'].get_last(), float), 'Incorrect metric type'
assert isinstance(exp['epoch/loss'].get_last(), float), 'Incorrect metric type'
assert isinstance(exp['batch/accuracy'].get_last(), float), 'Incorrect metric type'
assert isinstance(exp['batch/loss'].get_last(), float), 'Incorrect metric type'