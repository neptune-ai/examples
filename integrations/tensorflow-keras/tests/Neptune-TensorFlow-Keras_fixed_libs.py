# Neptune + TensorFlow / Keras

# Before we start

## Install dependencies

get_ipython().system(' pip install tensorflow==2.4.1 neptune-client==0.5.5  neptune-tensorflow-keras==0.9.1')

## Import libraries

import tensorflow as tf

## Define your model, data loaders and optimizer

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

optimizer = tf.keras.optimizers.SGD(lr=0.005, momentum=0.4,)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Quickstart

## Step 1: Initialize Neptune

import neptune.new as neptune

run = neptune.init(project='common/tf-keras-integration', api_token='ANONYMOUS')

## Step 2: Add NeptuneCallback to model.fit()

from neptune.new.integrations.tensorflow_keras import NeptuneCallback

neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

model.fit(x_train, y_train,
          epochs=5,
          batch_size=64,
          callbacks=[neptune_cbk])

## Step 3: Explore results in the Neptune UI

# tests
run.wait()

# check metrics
assert 0 <= run['metrics/batch/accuracy'].fetch_last() <= 1, 'Wrong values logged.'
assert 0 <= run['metrics/epoch/accuracy'].fetch_last() <= 1, 'Wrong values logged.'
assert 0 <= run['metrics/batch/loss'].fetch_last(), 'Wrong values logged.'
assert 0 <= run['metrics/epoch/loss'].fetch_last(), 'Wrong values logged.'

# More Options

## Log hardware consumption

get_ipython().system(' pip install psutil==5.6.6')

## Log hyperparameters

run_2 = neptune.init(project='common/tf-keras-integration', api_token='ANONYMOUS')

PARAMS = {'lr':0.005, 
          'momentum':0.9, 
          'epochs':10,
          'batch_size':32}

# log hyper-parameters
run_2['hyper-parameters'] = PARAMS

optimizer = tf.keras.optimizers.SGD(lr=PARAMS['lr'], momentum=PARAMS['momentum'])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

neptune_cbk_2 = NeptuneCallback(run=run_2, base_namespace='metrics')

model.fit(x_train, y_train,
          epochs=PARAMS['epochs'],
          batch_size=PARAMS['batch_size'],
          callbacks=[neptune_cbk_2])

## Log test sample images

for image in x_test[:100]:
    run_2['test/sample_images'].log(neptune.types.File.as_image(image))

## Log model weights

import glob

model.save('my_model')

run_2['my_model/saved_model'].upload('my_model/saved_model.pb')

for name in glob.glob('my_model/variables/*'):
    run_2[name].upload(name)

# Explore results in the Neptune UI

# tests
run_2.wait()

# check metrics
assert 0 <= run['metrics/batch/accuracy'].fetch_last() <= 1, 'Wrong values logged.'
assert 0 <= run['metrics/epoch/accuracy'].fetch_last() <= 1, 'Wrong values logged.'
assert 0 <= run['metrics/batch/loss'].fetch_last(), 'Wrong values logged.'
assert 0 <= run['metrics/epoch/loss'].fetch_last(), 'Wrong values logged.'