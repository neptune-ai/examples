import uuid
from datetime import datetime

import neptune
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Load sample dataset
bike_df = pd.read_csv("data/hour.csv")
bike_df["datetime"] = pd.to_datetime(bike_df["dteday"])
bike_df["datetime"] += pd.to_timedelta(bike_df.hr, unit="h")
bike_df.set_index("datetime", inplace=True)
bike_df = bike_df[
    [
        "season",
        "holiday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt",
    ]
]


# Define column mapping for Evidently
data_columns = ColumnMapping()
data_columns.numerical_features = ["weathersit", "temp", "atemp", "hum", "windspeed"]
data_columns.categorical_features = ["holiday", "workingday"]


# Define what to log
def eval_drift(reference, production, column_mapping):
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference, current_data=production, column_mapping=column_mapping
    )
    report = data_drift_report.as_dict()

    drifts = []

    for feature in column_mapping.numerical_features + column_mapping.categorical_features:
        drifts.append(
            (
                feature,
                report["metrics"][1]["result"]["drift_by_columns"][feature]["drift_score"],
            )
        )

    return drifts


# Define comparison windows
# Set reference dates
reference_dates = ("2011-01-01 00:00:00", "2011-06-30 23:00:00")

# Set experiment batches dates
experiment_batches = [
    ("2011-07-01 00:00:00", "2011-07-31 00:00:00"),
    ("2011-08-01 00:00:00", "2011-08-31 00:00:00"),
    ("2011-09-01 00:00:00", "2011-09-30 00:00:00"),
    ("2011-10-01 00:00:00", "2011-10-31 00:00:00"),
    ("2011-11-01 00:00:00", "2011-11-30 00:00:00"),
    ("2011-12-01 00:00:00", "2011-12-31 00:00:00"),
    ("2012-01-01 00:00:00", "2012-01-31 00:00:00"),
    ("2012-02-01 00:00:00", "2012-02-29 00:00:00"),
    ("2012-03-01 00:00:00", "2012-03-31 00:00:00"),
    ("2012-04-01 00:00:00", "2012-04-30 00:00:00"),
    ("2012-05-01 00:00:00", "2012-05-31 00:00:00"),
    ("2012-06-01 00:00:00", "2012-06-30 00:00:00"),
    ("2012-07-01 00:00:00", "2012-07-31 00:00:00"),
    ("2012-08-01 00:00:00", "2012-08-31 00:00:00"),
    ("2012-09-01 00:00:00", "2012-09-30 00:00:00"),
    ("2012-10-01 00:00:00", "2012-10-31 00:00:00"),
    ("2012-11-01 00:00:00", "2012-11-30 00:00:00"),
    ("2012-12-01 00:00:00", "2012-12-31 00:00:00"),
]


# (Neptune) Run and log drifts to Neptune

custom_run_id = str(uuid.uuid4())

for date in experiment_batches:
    with neptune.init_run(
        api_token=neptune.ANONYMOUS_API_TOKEN,
        project="common/evidently-support",
        custom_run_id=custom_run_id,  # Passing a custom run ID ensures that the metrics are logged to the same run.
        tags=["prod monitoring"],  # (optional) replace with your own
    ) as run:
        metrics = eval_drift(
            bike_df.loc[reference_dates[0] : reference_dates[1]],
            bike_df.loc[date[0] : date[1]],
            column_mapping=data_columns,
        )

        for feature in metrics:
            run["drift"][feature[0]].append(
                round(feature[1], 3),
                timestamp=datetime.strptime(date[0], "%Y-%m-%d %H:%M:%S").timestamp(),
            )
            # Passing a timestamp in the append() method lets you visualize the date in the x-axis of the charts
