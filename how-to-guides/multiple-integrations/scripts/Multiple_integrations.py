# This script uses Airflow to schedule an Optuna hyperparameter study on scikit-learn models
# and shows how to use Neptune's Airflow, Optuna, and scikit-learn integrations together.
#
# These integrations are just examples, and the same concepts can be applied to any of Neptune's other integrations.
# The docs for of all Neptune integrations are available here: https://docs.neptune.ai/integrations/
#
# This is an advanced example.
# We highly recommend going through the Airflow integration guide before proceeding:
# https://docs.neptune.ai/integrations/airflow/

import pickle as pkl
from datetime import datetime, timedelta

import optuna
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from neptune_airflow import NeptuneLogger
from neptune_optuna import NeptuneCallback
from neptune_sklearn import create_regressor_summary
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def prepare_dataset():
    data, target = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

    # Save data to disk
    pkl.dump(X_train, open("X_train.pkl", "wb"))
    pkl.dump(X_test, open("X_test.pkl", "wb"))
    pkl.dump(y_train, open("y_train.pkl", "wb"))
    pkl.dump(y_test, open("y_test.pkl", "wb"))


def objective(
    trial: optuna.Trial,
    logger: NeptuneLogger,
    **context,
):
    """Objective function for Optuna, with Neptune's scikit-learn integration to log model metadata"""

    # Load data from disk
    X_train = pkl.load(open("X_train.pkl", "rb"))
    X_test = pkl.load(open("X_test.pkl", "rb"))
    y_train = pkl.load(open("y_train.pkl", "rb"))
    y_test = pkl.load(open("y_test.pkl", "rb"))

    # Hyperparameter space for Optuna
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 2, 64),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_samples_split": trial.suggest_int("min_samples_split", 3, 10),
    }

    # Train the model
    model = RandomForestRegressor(**param)
    model.fit(X_train, y_train)

    # Fetch the current run
    with logger.get_run_from_context(context=context, log_context=True) as run:
        # Log model summary for each trial under the "sklearn" namespace
        # Neptune+scikit-learn integration docs: https://docs.neptune.ai/integrations/sklearn/
        run[f"sklearn/model_summary_{trial.number}"] = create_regressor_summary(
            model, X_train, X_test, y_train, y_test
        )
        run.wait()

        # Fetch objective score from the run
        score = run[f"sklearn/model_summary_{trial.number}/test/scores/mean_absolute_error"].fetch()

        return score


def train_model_with_hpo(logger: NeptuneLogger, **context):
    # Get run handler for the context
    with logger.get_task_handler_from_context(
        context=context,
        log_context=True,  # This will log the context under the <TASK_ID> namespace
    ) as handler:
        # Fetch run from handler
        run = handler.get_root_object()

        # Create Optuna study
        study = optuna.create_study(direction="minimize")

        # Initialize Neptune's callback for Optuna
        # Neptune+Optuna integration docs: https://docs.neptune.ai/integrations/optuna/
        neptune_optuna_callback = NeptuneCallback(
            run,
            base_namespace="optuna",  # All Optuna metadata will be logged in the "optuna" namespace
        )

        # Run the Optuna study
        study.optimize(
            lambda trial: objective(trial, logger, **context),
            n_trials=3,
            callbacks=[neptune_optuna_callback],
        )


def get_neptune_token_from_variable() -> dict[str, str]:
    """Reads NEPTUNE_API_TOKEN and NEPTUNE_PROJECT from Airflow variables.

    Returns:
        dict[str,str]: A dict containing the NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
    """
    return {
        "api_token": Variable.get("NEPTUNE_API_TOKEN", None),
        "project": Variable.get(
            key="NEPTUNE_PROJECT",
            default_var="common/multiple-integrations",  # remove or replace with your own default
        ),
    }


def on_failure_callback(context):
    # We want the Python script to error if any task fails.
    exit(1)


with DAG(
    dag_id="example_dag",
    description="Dataset preparation and HPO",
    tags=["neptune", "airflow", "optuna", "sklearn"],
    schedule="@daily",
    start_date=datetime.today() - timedelta(days=1),
    catchup=False,
    default_args={
        "on_failure_callback": on_failure_callback,
    },
) as dag:

    @task(task_id="data")
    def data_task():
        return prepare_dataset()  # This task is not logged to Neptune

    @task(task_id="train")
    def train_task(**context):
        # Initialize Neptune logger for Airflow
        logger = NeptuneLogger(
            **get_neptune_token_from_variable(),
            tags=["script", "airflow", "sklearn", "optuna"],
        )
        return train_model_with_hpo(logger, **context)  # This task is logged to Neptune

    data_task() >> train_task()

if __name__ == "__main__":
    dag.test()
