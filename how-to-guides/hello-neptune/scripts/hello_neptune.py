import neptune

# Initialize Neptune and create a new run
run = neptune.init_run(
    project="common/quickstarts",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["quickstart", "script"],
    dependencies="infer",  # to infer dependencies. You can also pass the path to the requirements.txt file
)

# log single value
run["seed"] = 0.42

# log series of values
from random import random

epochs = 10
offset = random() / 5

for epoch in range(epochs):
    acc = 1 - 2**-epoch - random() / (epoch + 1) - offset
    loss = 2**-epoch + random() / (epoch + 1) + offset

    run["accuracy"].append(acc)
    run["loss"].append(loss)

# Upload single image to Neptune
run["single_image"].upload("Lenna_test_image.png")  # You can upload native images as-is

# Download MNIST dataset
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Upload a series of images to Neptune
from neptune.types import File

for i in range(10):
    run["image_series"].append(
        File.as_image(
            train_images[i] / 255
        ),  # You can upload arrays as images using Neptune's File.as_image() method
        name=f"{train_labels[i]}",
    )

# Save the run ID to resume the run later
run_id = run["sys/id"].fetch()

# Stop logging
run.stop()

# Reinitialize an already logged run
run = neptune.init_run(
    project="common/quickstarts",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    with_id=run_id,  # ID of the run you want to re-initialize
    mode="read-only",  # To prevent accidental overwrite of already logged data
)

# Download metadata from reinitialized run
print(f"Logged seed: {run['seed'].fetch()}")
print(f"Logged accuracies:\n{run['accuracy'].fetch_values()}")
run["single_image"].download("downloaded_single_image.png")
print("Image downloaded to downloaded_single_image.png")
run["image_series"].download("downloaded_image_series")
print("Image series downloaded to downloaded_image_series folder")

# Stop the run
run.stop()
