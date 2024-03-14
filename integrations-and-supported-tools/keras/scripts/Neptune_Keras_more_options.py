# Import libraries
import neptune
import tensorflow as tf
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File
from tensorflow.keras.callbacks import ModelCheckpoint

# Prepare dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build and compile model
params = {"lr": 0.005, "momentum": 0.7, "epochs": 15, "batch_size": 256}

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
    ]
)

optimizer = tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=params["momentum"])

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Initialize Keras' ModelCheckpoint
checkpoint_cbk = ModelCheckpoint(
    "checkpoints/ep{epoch:02d}-acc{accuracy:.3f}.keras",
    save_best_only=False,
    save_weights_only=False,
    save_freq="epoch",
)

# (Neptune) Initialize run
run = neptune.init_run(
    project="common/tf-keras-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["script", "more options"],
)

# (Neptune) log hyper-parameters
run["hyper-parameters"] = params

# (Neptune) Initialize NeptuneCallback to log metrics during training
neptune_cbk = NeptuneCallback(
    run=run,
    log_on_batch=True,
    log_model_diagram=False,  # Requires pydot to be installed
)

# Fit model with callbacks
model.fit(
    x_train,
    y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[neptune_cbk, checkpoint_cbk],
)

# (Neptune) Upload model checkpoints
run["checkpoints"].upload_files("checkpoints")

# (Neptune) Upload final model
model.save("my_model.keras")

run["saved_model"].upload("my_model.keras")

# (Neptune) log test images with prediction
for image, label in zip(x_test[:10], y_test[:10]):
    prediction = model.predict(image[None], verbose=0)
    predicted = prediction.argmax()
    desc = f"label : {label} | predicted : {predicted}"
    run["visualization/test_prediction"].append(File.as_image(image), description=desc)
