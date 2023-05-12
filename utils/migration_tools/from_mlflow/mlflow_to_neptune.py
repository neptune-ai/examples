import mlflow
import neptune
import pandas as pd
from neptune import management

ARTIFACT_BUCKET = "s3://<YOUR BUCKET>/"
MLFLOW_TRACKING_URI = "<YOUR TRACKING URI>"
NEPTUNE_WORKSPACE = "<YOUR WORKSPACE>"
NEPTUNE_API_TOKEN = "<YOUR NEPTUNE API TOKEN>"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

### Create neptune project per mlflow experiment
# TODO: Give users an option to export al MLflow runs to a single Neptune project

all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]

for exp_id in all_experiments:
    management.create_project(
        workspace=NEPTUNE_WORKSPACE, name=f"mlflow-{exp_id}", api_token=NEPTUNE_API_TOKEN
    )

### Get all runs in mlflow

all_runs = mlflow.search_runs(experiment_ids=all_experiments, run_view_type=ViewType.ALL)

client = mlflow.tracking.MlflowClient()

columns = list(all_runs.columns)

for index, row in all_runs.iterrows():
    ### Prepare data to be pushed to neptune per run

    params = []
    metrics = []
    tags = []

    data_rep = {
        "run_id": row["run_id"],
        "experiment_id": f'mlflow-{row["experiment_id"]}',
        "artifact_uri": row["artifact_uri"].replace("mlflow-artifacts:/", ARTIFACT_BUCKET),
    }

    for data in zip(columns, row.to_list()):
        col_name = data[0]
        value = data[1]

        if not pd.isna(value):
            if col_name.startswith("params."):
                params.append([col_name.replace("params.", ""), value])
            elif col_name.startswith("metrics."):
                metric_key = col_name.replace("metrics.", "")
                metrics_list = [
                    metric.value
                    for metric in client.get_metric_history(data_rep["run_id"], metric_key)
                ]
                metrics.append([metric_key, metrics_list])
            elif col_name.startswith("tags."):
                tags.append([col_name.replace("tags.", ""), value])

    data_rep["params"] = params
    data_rep["metrics"] = metrics
    data_rep["tags"] = tags

    ### Push all metadata except artifacts to neptune.

    run = neptune.init_run(
        project=f'{NEPTUNE_WORKSPACE}/{data_rep["experiment_id"]}',
        custom_run_id=data_rep["run_id"],
        mode="sync",
        api_token=NEPTUNE_API_TOKEN,
    )

    for param in data_rep["params"]:
        run["parameters"][param[0]] = param[1]
    for metrics in data_rep["metrics"]:
        run[f"metrics/{metrics[0]}"].extend(metrics[1])
    for tag in data_rep["tags"]:
        run["sys/tags"].add(
            f"{tag[0]}={tag[1]}"
        )  ### Tags are concatenated with = sign to represent mlflow tagging, but you could treat tags as a separate field in parameters to be searcheable

    run.stop()
