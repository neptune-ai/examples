import io

import neptune
import numpy as np
import requests
import tensorflow as tf

run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,  # replace with your own token
    project="common/tensorflow-support",  # replace with your own project
    tags=["script"],
)


response = requests.get("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
with open("mnist.npz", "wb") as f:
    f.write(response.content)

# (Neptune) Track and version data files used for training
run["datasets/version"].track_files("mnist.npz")

with np.load("mnist.npz") as data:
    train_examples = data["x_train"]
    train_labels = data["y_train"]
    test_examples = data["x_test"]
    test_labels = data["y_test"]

# Parameters for training
params = {
    "batch_size": 512,
    "shuffle_buffer_size": 100,
    "lr": 0.01,
    "num_epochs": 15,
    "num_visualization_examples": 10,
}

# (Neptune) Log training parameters
run["training/model/params"] = params


# Normalize data for training
train_examples = train_examples.astype(np.float32)
test_examples = test_examples.astype(np.float32)

# Prepare data for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))


train_dataset = train_dataset.shuffle(params["shuffle_buffer_size"]).batch(params["batch_size"])
test_dataset = test_dataset.batch(params["batch_size"])

# Prepare model
# Model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

# Loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Optimizer
optimizer = tf.keras.optimizers.Adam(params["lr"])


# (Neptune) Log model summary
with io.StringIO() as s:
    model.summary(print_fn=lambda x, **kwargs: s.write(x + "\n"))
    model_summary = s.getvalue()

# (Neptune) Log model summary
run["training/model/summary"] = model_summary


# Helper functions for training loop
def loss_and_preds(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_), y_


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

    # (Neptune) Log metrics for the epoch
    # Train metrics
    run["training/train/loss"].append(epoch_loss_avg.result())
    run["training/train/accuracy"].append(epoch_accuracy.result())

    # (Neptune) Log test metrics
    test_loss, test_preds = loss_and_preds(model, test_examples, test_labels, False)
    run["training/test/loss"].append(test_loss)
    test_acc = epoch_accuracy(test_labels, test_preds)
    run["training/test/accuracy"].append(test_acc)

    # (Neptune) Log test prediction
    for idx in range(params["num_visualization_examples"]):
        np_image = test_examples[idx].reshape(28, 28)
        image = neptune.types.File.as_image(np_image)
        pred_label = test_preds[idx].numpy().argmax()
        true_label = test_labels[idx]
        run[f"training/visualization/epoch_{epoch}"].append(
            image, description=f"pred={pred_label} | actual={true_label}"
        )

    if epoch % 5 == 0 or epoch == (params["num_epochs"] - 1):
        print(
            "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, epoch_loss_avg.result(), epoch_accuracy.result()
            )
        )

# Saves model artifacts to "weights" folder
model.save("weights.keras")
# (Neptune) Log model artifacts
run["training/model/weights"].upload("weights.keras")
