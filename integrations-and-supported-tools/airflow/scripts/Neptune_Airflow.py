from datetime import datetime, timedelta

import tensorflow as tf
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
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
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # (Neptune) log metrics during training using the above handler
        neptune_cbk = NeptuneCallback(run=handler)
        model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[neptune_cbk])

        model.save("model_checkpoint.keras")

        # Upload checkpoint to Neptune if tasks don't run on same machine
        # run = handler.get_root_object()
        # run["model_checkpoint"].upload("model_checkpoint.keras")
        # run.wait()


def evaluate_model(logger: NeptuneLogger, **context):
    # (Neptune) This will be logged relative to `task_name/{namespace}`
    with logger.get_task_handler_from_context(context=context) as handler:
        # Download model checkpoint from Neptune if the tasks don't share the same file system
        # run = handler.get_root_object()
        # run["model_checkpoint"].download()
        model = tf.keras.models.load_model("model_checkpoint.keras")
        _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test / 255.0

        for image, label in zip(x_test[:10], y_test[:10]):
            prediction = model.predict(image[None], verbose=0)
            predicted = prediction.argmax()
            desc = f"label : {label} | predicted : {predicted}"
            # (Neptune) log predictions using the above handler.
            handler["visualization/test_prediction"].append(File.as_image(image), description=desc)


def get_neptune_token_from_variable() -> "dict[str, str]":
    """Reads NEPTUNE_API_TOKEN and NEPTUNE_PROJECT from Airflow variables.

    Returns:
        dict[str,str]: A dict containing the NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
    """
    return {
        "api_token": Variable.get("NEPTUNE_API_TOKEN", None),
        "project": Variable.get(
            key="NEPTUNE_PROJECT",
            default_var="common/airflow-integration",  # remove or replace with your own default
        ),
    }


def on_failure_callback(context):
    # We want the Python script to
    # error if any task fails.
    exit(1)


with DAG(
    dag_id="test_dag",
    description="test_description",
    tags=["neptune", "tensorflow"],
    schedule="@daily",
    start_date=datetime.today() - timedelta(days=1),
    catchup=False,
    default_args={
        "on_failure_callback": on_failure_callback,
    },
) as dag:

    @task(task_id="data")
    def data_task(**context):
        # (Neptune) We recommend passing the Neptune API token and project name using
        #           Airflow variables, especially when tasks are run on different
        #           machines.
        logger = NeptuneLogger(**get_neptune_token_from_variable())
        return data_details(logger, **context)

    @task(task_id="train")
    def train_task(**context):
        logger = NeptuneLogger(**get_neptune_token_from_variable())
        return train_model(logger, **context)

    @task(task_id="evaluate")
    def evaluate_task(**context):
        logger = NeptuneLogger(**get_neptune_token_from_variable())
        return evaluate_model(logger, **context)

    data_task() >> train_task() >> evaluate_task()

if __name__ == "__main__":
    dag.test()
