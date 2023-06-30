import io

import neptune
import numpy as np
import requests
import tensorflow as tf
from neptune_tensorboard import enable_tensorboard_logging

# (Neptune) Create a run for logging.
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/tensorboard-integration",  # replace with your own
    tags=["script", "sync"],  # optional
)

# (Neptune) Enable tensorboard logger to also log to the Neptune `run`
# Calling `enable_tensorboard_logging` also works with SummaryWriter from PyTorch and tensorboardX
# and also `Tensorboard` callback for Keras.
# NOTE: This will log to both tensorboard directory and the Neptune run.
enable_tensorboard_logging(run)

writer = tf.summary.create_file_writer("logs")
writer.set_as_default(0)

response = requests.get("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
with open("mnist.npz", "wb") as f:
    f.write(response.content)


with np.load("mnist.npz") as data:
    train_examples = data["x_train"]
    train_labels = data["y_train"]
    test_examples = data["x_test"]
    test_labels = data["y_test"]

# Parameters for training
params = {
    "batch_size": 1024,
    "shuffle_buffer_size": 100,
    "lr": 0.001,
    "num_epochs": 5,
    "num_visualization_examples": 5,
}

# Log training parameters
for name, value in params.items():
    tf.summary.scalar(name, value)


# Normalize data for training
def normalize_img(image):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0


train_examples = normalize_img(train_examples)
test_examples = normalize_img(test_examples)

# Prepare data for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

train_dataset = train_dataset.shuffle(params["shuffle_buffer_size"]).batch(params["batch_size"])
test_dataset = test_dataset.batch(params["batch_size"])

# Prepare model
# Model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

# Loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Optimizer
optimizer = tf.keras.optimizers.Adam(params["lr"])


# Log model summary
with io.StringIO() as s:
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    model_summary = s.getvalue()

tf.summary.text("model_summary", model_summary)


# Helper functions for training loop
def loss_and_preds(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_pred = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_pred), y_pred


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value, _ = loss_and_preds(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Training Loop
for epoch in range(params["num_epochs"]):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, model(x, training=True))

    # Log metrics for the epoch
    # Train metrics
    tf.summary.scalar("loss", epoch_loss_avg.result())
    tf.summary.scalar("accuracy", epoch_accuracy.result())

    # Log test metrics
    test_loss, test_preds = loss_and_preds(model, test_examples, test_labels, False)
    test_acc = epoch_accuracy(test_labels, test_preds)
    tf.summary.scalar("test_loss", test_loss)
    tf.summary.scalar("test_accuracy", test_acc)

    # Log test prediction
    for idx in range(params["num_visualization_examples"]):
        np_image = test_examples[idx].numpy().reshape(1, 28, 28, 1)
        pred_label = test_preds[idx].numpy().argmax()
        true_label = test_labels[idx]
        tf.summary.image(f"epoch-{epoch}_pred-{pred_label}_actual-{true_label}", np_image)

    if epoch % 5 == 0 or epoch == (params["num_epochs"] - 1):
        print(
            "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, epoch_loss_avg.result(), epoch_accuracy.result()
            )
        )
