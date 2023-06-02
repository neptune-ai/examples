import neptune

# Initialize Neptune and create a new run
run = neptune.init_run(
    project="common/quickstarts",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["quickstart", "script"],
)

# log single value
run["seed"] = 0.42

# log series of values
from random import random
from time import sleep

epochs = 10
offset = random() / 5

for epoch in range(epochs):
    sleep(0.2)  # to see logging live
    acc = 1 - 2**-epoch - random() / (epoch + 1) - offset
    loss = 2**-epoch + random() / (epoch + 1) + offset

    run["accuracy"].append(acc)
    run["loss"].append(loss)

# Upload single image to Neptune
run["single_image"].upload("Lenna_test_image.png")  # Native images can be upload as-is

# Download MNIST dataset
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Upload a series of images to Neptune
from neptune.types import File

for i in range(10):
    run["image_series"].append(
        File.as_image(
            train_X[i] / 255
        ),  # Arrays can be uploaded as images using Neptune's `File.as_image()` method
        name=f"{train_y[i]}",
    )

# Stop logging
run.stop()
