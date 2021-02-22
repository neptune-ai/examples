# Neptune API tour
# 

# Introduction

# Setup

get_ipython().system(' pip install --quiet git+https://github.com/neptune-ai/neptune-client.git@alpha')

# Initialize Neptune

import neptune.alpha as neptune

exp = neptune.init(project='common/colab-test-run',
                   api_token='ANONYMOUS')

# Basic Example

params = {'learning_rate': 0.1}

# log params
exp['parameters'] = params

# log name and append tags
exp["sys/name"] = 'basic-colab-example'
exp["sys/tags"].add(['colab', 'intro'])

# log loss during training
for epoch in range(100):
    exp["train/loss"].log(0.99 ** epoch)

# log train and validation scores
exp['train/accuracy'] = 0.95
exp['valid/accuracy'] = 0.93

# tests
exp.wait()

# check train/loss
assert exp['train/loss'].get_last() < 1.0, 'Wrong loss values logged.'

# check tags
all_tags = ['colab', 'intro']
assert set(exp["sys/tags"].get()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, exp["sys/tags"].get())

# check scores
tr = 0.95
va = 0.93

assert exp['train/accuracy'].get() == tr, 'Expected: {}, Actual: {}'.format(tr, exp['train/accuracy'].get())
assert exp['valid/accuracy'].get() == va, 'Expected: {}, Actual: {}'.format(va, exp['valid/accuracy'].get())

# Keras classification example [Advanced]

get_ipython().system(' pip install --quiet tensorflow==2.3.1 scikit-plot==0.3.7')

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
            exp['batch/{}'.format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            exp['epoch/{}'.format(log_name)].log(log_value)

EPOCH_NR = 5
BATCH_SIZE = 32

exp = neptune.init(project='common/colab-test-run',
                   api_token='ANONYMOUS')

# log params
exp['parameters/epoch_nr'] = EPOCH_NR
exp['parameters/batch_size'] = BATCH_SIZE

# log name and append tag
exp["sys/name"] = 'keras-metrics'
exp["sys/tags"].add('advanced')

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

exp['test/f1'] = f1_score(y_test, y_test_pred_class, average='micro')

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_test, y_test_pred_class, ax=ax)
exp['diagnostic_charts'].log(neptune.types.Image(fig))

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y_test, y_test_pred, ax=ax)
exp['diagnostic_charts'].log(neptune.types.Image(fig))

model.save('my_model.h5')
exp["model"].save('my_model.h5')

# tests
exp.wait()

# check train/loss
assert exp['epoch/loss'].get_last() < 1.0, 'Wrong loss values logged.'

# check tags
all_tags = ['advanced']
assert set(exp["sys/tags"].get()) == set(all_tags), 'Expected: {}, Actual: {}'.format(all_tags, exp["sys/tags"].get())

# check params
batch_size = 32
epoch_nr = 5

assert exp['parameters/batch_size'].get() == batch_size, 'Expected: {}, Actual: {}'.format(batch_size, exp['parameters/batch_size'].get())
assert exp['parameters/epoch_nr'].get() == epoch_nr, 'Expected: {}, Actual: {}'.format(epoch_nr, exp['parameters/epoch_nr'].get())

# Access data you logged programatically 

# Getting the project's leaderboard

my_project = neptune.get_project('common/colab-test-run')
exp_df = my_project.get_experiments_table(tag=['advanced']).as_pandas()
exp_df.head()

# Getting the experiment's metadata

exp = neptune.init(project='common/colab-test-run' ,experiment='COL-7')

batch_size = exp["parameters/batch_size"].get()
last_batch_acc = exp['batch/accuracy'].get_last()
print('batch_size: {}'.format(batch_size))
print('last_batch_acc: {}'.format(last_batch_acc))