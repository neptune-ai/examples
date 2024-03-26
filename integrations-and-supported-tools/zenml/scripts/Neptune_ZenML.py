from typing import Tuple

import pandas as pd
from neptune.types import File
from neptune.utils import stringify_unsupported
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from zenml import get_step_context, pipeline, step
from zenml.client import Client
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
from zenml.integrations.neptune.flavors import NeptuneExperimentTrackerSettings

client = Client()

# Get neptune_tracker component from stack
neptune_tracker = client.get_stack_component(
    component_type="experiment_tracker", name_id_or_prefix="neptune_tracker"
).name

# Add tags to Neptune run
neptune_settings = NeptuneExperimentTrackerSettings(tags={"regression", "script", "sklearn"})


@step(
    experiment_tracker=neptune_tracker,
    settings={"experiment_tracker.neptune": neptune_settings},
    enable_cache=False,
)
def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # Get neptune run
    neptune_run = get_neptune_run()

    # Log pipeline and step metadata to run
    context = get_step_context()
    neptune_run["pipeline"] = stringify_unsupported(context.pipeline_run.get_metadata().dict())
    neptune_run[f"steps/{context.step_name}"] = stringify_unsupported(
        context.step_run.get_metadata().dict()
    )

    data = fetch_california_housing(as_frame=True).frame

    # Log dataset to run
    neptune_run["dataset"].upload(File.as_html(data.sample(100)))

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["MedHouseVal"]), data["MedHouseVal"]
    )

    return X_train, X_test, y_train, y_test


@step(
    experiment_tracker=neptune_tracker,
    settings={"experiment_tracker.neptune": neptune_settings},
    enable_cache=False,
)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:

    # Get neptune run
    neptune_run = get_neptune_run()

    # Log step metadata to run
    context = get_step_context()
    neptune_run[f"steps/{context.step_name}"] = stringify_unsupported(
        context.step_run.get_metadata().dict()
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Upload model
    neptune_run["model"].upload(File.as_pickle(model))

    return model


@step(
    experiment_tracker=neptune_tracker,
    settings={"experiment_tracker.neptune": neptune_settings},
    enable_cache=False,
)
def evaluate_model(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:

    # Get neptune run
    neptune_run = get_neptune_run()

    # Log step metadata to run
    context = get_step_context()
    neptune_run[f"steps/{context.step_name}"] = stringify_unsupported(
        context.step_run.get_metadata().dict()
    )

    # Make predictions
    preds = model.predict(X_test)

    # Log metrics
    neptune_run["metrics"] = {
        "r2_score": r2_score(y_test, preds),
        "mape": mean_absolute_percentage_error(y_test, preds),
    }


@pipeline
def neptune_example_pipeline():
    """
    Link all the steps artifacts together
    """
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    neptune_example_pipeline()
