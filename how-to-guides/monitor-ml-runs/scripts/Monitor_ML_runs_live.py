from tensorflow import keras
import neptune.new as neptune

run = neptune.init(project="common/quickstarts", api_token="ANONYMOUS")

PARAMS = {
    "epoch_nr": 10,
    "batch_size": 256,
    "lr": 0.005,
    "momentum": 0.4,
    "use_nesterov": True,
    "unit_nr": 256,
    "dropout": 0.05,
}

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(PARAMS["unit_nr"], activation=keras.activations.relu),
        keras.layers.Dropout(PARAMS["dropout"]),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ]
)

optimizer = keras.optimizers.SGD(
    lr=PARAMS["lr"],
    momentum=PARAMS["momentum"],
    nesterov=PARAMS["use_nesterov"],
)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# log metrics during training
class NeptuneLogger(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            run["batch/{}".format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            run["epoch/{}".format(log_name)].log(log_value)


model.fit(
    x_train,
    y_train,
    epochs=PARAMS["epoch_nr"],
    batch_size=PARAMS["batch_size"],
    callbacks=[NeptuneLogger()],
)
