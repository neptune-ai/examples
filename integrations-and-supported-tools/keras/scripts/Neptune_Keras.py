import neptune
import tensorflow as tf
from neptune.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init_run(project="common/tf-keras-integration", api_token=neptune.ANONYMOUS_API_TOKEN)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
    ]
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.005,
    momentum=0.4,
)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# (Neptune) log metrics during training
neptune_cbk = NeptuneCallback(run=run)
model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[neptune_cbk])
