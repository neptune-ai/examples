from datetime import datetime, timedelta

import tensorflow as tf
from airflow import DAG
from airflow.decorators import task
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File
from neptune_airflow import NeptuneLogger


def data_details(logger: NeptuneLogger, **context):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0

    # (Neptune) This will be logged relative to `task_name/{namespace}` (including the context)
    with logger.get_task_handler_from_context(context=context, log_context=True) as handler:
        handler["num_train"] = len(y_train)
        handler["num_test"] = len(y_test)

    # (Neptune) This will be logged relative to root namespace (including the context)
    with logger.get_run_from_context(context=context, log_context=True) as run:
        run["data/train_sample"].upload(File.as_image(x_train[0]))


def train_model(logger: NeptuneLogger, **context):
    with logger.get_task_handler_from_context(context=context) as handler:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train / 255.0

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

        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # (Neptune) log metrics during training using the above handler
        neptune_cbk = NeptuneCallback(run=handler)
        model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[neptune_cbk])

        model.save("my_model.h5")

        # If task don't run on same machine then upload checkpoint.
        # run = handler.get_root_object()
        # run["model_checkpoint/checkpoint"].upload_files("my_model.h5")


def evaluate_model(logger: NeptuneLogger, **context):
    # (Neptune) This will be logged relative to `task_name/{namespace}`
    with logger.get_task_handler_from_context(context=context) as handler:
        # if the tasks don't share the same file system
        # run["model_checkpoint/checkpoint/my_model.h5"].download()

        model = tf.keras.models.load_model("my_model.h5")
        _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test / 255.0

        for image, label in zip(x_test[:10], y_test[:10]):
            prediction = model.predict(image[None], verbose=0)
            predicted = prediction.argmax()
            desc = f"label : {label} | predicted : {predicted}"
            # (Neptune) log predictions using the above handler.
            handler["visualization/test_prediction"].append(File.as_image(image), description=desc)


with DAG(
    dag_id="test_dag",
    description="test_description",
    tags=["neptune", "tensorflow"],
    schedule="@daily",
    start_date=datetime.today() - timedelta(days=1),
    catchup=False,
) as dag:

    @task(task_id="data")
    def data_task(**context):
        logger = NeptuneLogger()
        return data_details(logger, **context)

    @task(task_id="train")
    def train_task(**context):
        logger = NeptuneLogger()
        return train_model(logger, **context)

    @task(task_id="evaluate")
    def evaluate_task(**context):
        logger = NeptuneLogger()
        return evaluate_model(logger, **context)

    data_task() >> train_task() >> evaluate_task()

dag.test()
