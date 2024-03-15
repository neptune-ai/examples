import neptune
import tensorflow as tf
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File

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

run = neptune.init_run(
    project="common/tf-keras-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["script", "more options"],
)

params = {"lr": 0.005, "momentum": 0.9, "epochs": 15, "batch_size": 256}

# (Neptune) log hyper-parameters
run["hyper-parameters"] = params

optimizer = tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=params["momentum"])

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# (Neptune) log metrics during training
neptune_cbk = NeptuneCallback(
    run=run,
    log_on_batch=True,
    log_model_diagram=False,  # Requires pydot to be installed
)

model.fit(
    x_train,
    y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[neptune_cbk],
)

# (Neptune) log test images with prediction
for image, label in zip(x_test[:10], y_test[:10]):
    prediction = model.predict(image[None], verbose=0)
    predicted = prediction.argmax()
    desc = f"label : {label} | predicted : {predicted}"
    run["visualization/test_prediction"].append(File.as_image(image), description=desc)


# (Neptune) Upload model weights
model.save("my_model.keras")

run["model_checkpoint"].upload("my_model.keras")
