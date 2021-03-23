# Neptune API tour
# 

# Introduction

# Setup

# Initialize Neptune

import neptune.new as neptune

run = neptune.init(project='common/colab-test-run',
                   api_token='ANONYMOUS')

# Basic Example

params = {'learning_rate': 0.1}

# log params
run['parameters'] = params

# log name and append tags
run["sys/name"] = 'basic-colab-example'
run["sys/tags"].add(['colab', 'intro'])

# log loss during training
for epoch in range(100):
    run["train/loss"].log(0.99 ** epoch)

# log train and validation scores
run['train/accuracy'] = 0.95
run['valid/accuracy'] = 0.93

# Keras classification example [Advanced]

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import Callback

class NeptuneLogger(Callback):
    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            run['batch/{}'.format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            run['epoch/{}'.format(log_name)].log(log_value)

EPOCH_NR = 5
BATCH_SIZE = 32

run = neptune.init(project='common/colab-test-run',
                   api_token='ANONYMOUS')

# log params
run['parameters/epoch_nr'] = EPOCH_NR
run['parameters/batch_size'] = BATCH_SIZE

# log name and append tag
run["sys/name"] = 'keras-metrics'
run["sys/tags"].add('advanced')

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=EPOCH_NR,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=[NeptuneLogger()])

import numpy as np

y_test_pred = np.asarray(model.predict(x_test))
y_test_pred_class = np.argmax(y_test_pred, axis=1)

from sklearn.metrics import f1_score

run['test/f1'] = f1_score(y_test, y_test_pred_class, average='micro')

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_test, y_test_pred_class, ax=ax)
run['diagnostic_charts'].log(neptune.types.File.as_image(fig))

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y_test, y_test_pred, ax=ax)
run['diagnostic_charts'].log(neptune.types.File.as_image(fig))

model.save('my_model.h5')
run["model"].upload('my_model.h5')

# Access data you logged programatically 

# Getting the project's leaderboard

my_project = neptune.get_project(name='common/colab-test-run', api_token='ANONYMOUS')
run_df = my_project.fetch_runs_table(tag=['advanced']).to_pandas()
run_df.head()

# Getting the run's metadata

run = neptune.init(project='common/colab-test-run', api_token='ANONYMOUS', run='COL-7')

batch_size = run["parameters/batch_size"].fetch()
last_batch_acc = run['batch/accuracy'].fetch_last()
print('batch_size: {}'.format(batch_size))
print('last_batch_acc: {}'.format(last_batch_acc))