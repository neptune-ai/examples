import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import tensorflow as tf
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.metrics import f1_score

run = neptune.init(project="common/colab-test-run", api_token="ANONYMOUS")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


class NeptuneLogger(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        for log_name, log_value in logs.items():
            run["batch/{}".format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for log_name, log_value in logs.items():
            run["epoch/{}".format(log_name)].log(log_value)


EPOCH_NR = 5
BATCH_SIZE = 32

# log params
run["parameters/epoch_nr"] = EPOCH_NR
run["parameters/batch_size"] = BATCH_SIZE

# log name and append tag
run["sys/name"] = "keras-metrics"
run["sys/tags"].add("advanced")

history = model.fit(
    x=x_train,
    y=y_train,
    epochs=EPOCH_NR,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=[NeptuneLogger()],
)

# log metric
y_test_pred = np.asarray(model.predict(x_test))
y_test_pred_class = np.argmax(y_test_pred, axis=1)

run["test/f1"] = f1_score(y_test, y_test_pred_class, average="micro")

# log diagnostic charts
fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_test, y_test_pred_class, ax=ax)
run["diagnostic_charts"].log(neptune.types.File.as_image(fig))

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y_test, y_test_pred, ax=ax)
run["diagnostic_charts"].log(neptune.types.File.as_image(fig))

# log model weights
model.save("my_model.h5")
run["model"].upload("my_model.h5")
