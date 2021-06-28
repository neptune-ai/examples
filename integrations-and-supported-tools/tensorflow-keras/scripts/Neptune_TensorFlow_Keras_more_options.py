import glob
import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

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

run = neptune.init(project="common/tf-keras-integration", api_token="ANONYMOUS")

PARAMS = {"lr": 0.005, "momentum": 0.9, "epochs": 10, "batch_size": 32}

# log hyper-parameters
run["hyper-parameters"] = PARAMS

optimizer = tf.keras.optimizers.SGD(
    learning_rate=PARAMS["lr"], momentum=PARAMS["momentum"]
)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# log metrics during training
neptune_cbk = NeptuneCallback(run=run, base_namespace="metrics")
model.fit(
    x_train,
    y_train,
    epochs=PARAMS["epochs"],
    batch_size=PARAMS["batch_size"],
    callbacks=[neptune_cbk],
)

# log images
for image in x_test[:100]:
    run["test/sample_images"].log(neptune.types.File.as_image(image))

# log model
model.save("my_model")

run["my_model/saved_model"].upload("my_model/saved_model.pb")
for name in glob.glob("my_model/variables/*"):
    run[name].upload(name)
