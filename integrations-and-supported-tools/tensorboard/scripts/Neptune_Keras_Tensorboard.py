import datetime

import neptune
import tensorflow as tf
from neptune_tensorboard import enable_tensorboard_logging

run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/quickstarts",  # replace with your own
)

enable_tensorboard_logging(run)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28), name="layers_flatten"),
            tf.keras.layers.Dense(512, activation="relu", name="layers_dense"),
            tf.keras.layers.Dropout(0.2, name="layers_dropout"),
            tf.keras.layers.Dense(10, activation="softmax", name="layers_dense_2"),
        ]
    )


model = create_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

log_dir = "logs/keras/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(
    x=x_train,
    y=y_train,
    epochs=2,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)
