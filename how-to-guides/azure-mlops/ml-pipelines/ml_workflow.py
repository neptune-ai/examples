import hashlib
import os
import time

import azureml.core
from azure.ai.ml import Input, MLClient
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core import (
    ComputeTarget,
    Dataset,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)


# load workspace
workspace = Workspace.from_config()
print(
    "Workspace name: " + workspace.name,
    "Azure region: " + workspace.location,
    "Subscription id: " + workspace.subscription_id,
    "Resource group: " + workspace.resource_group,
    sep="\n",
)

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()


# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)


# choose a name for your cluster
def get_compute_cluster(cluster_name):
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
        print("Found existing compute target")
    except ComputeTargetException:
        print("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_NC6", max_nodes=4)

        # create the cluster
        compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)

        # can poll for a minimum number of nodes and for a specific timeout.
        # if no min node count is provided it uses the scale settings for the cluster
        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20
        )

    # use get_status() to get a detailed status for the current cluster.
    print(compute_target.get_status().serialize())
    return compute_target


compute_target = get_compute_cluster("cpu-cluster-2")


# Get the dataset
data = Input(
    path="https://raw.githubusercontent.com/neptune-ai/examples/main/use-cases/time-series-forecasting/walmart-sales/dataset/aggregate_data.csv"
)
dataset = Dataset.Tabular.from_delimited_files(path=data["path"])


# create an ML experiment
exp = Experiment(workspace=workspace, name="time-series")

# create a directory
script_folder = "./src"
os.makedirs(script_folder, exist_ok=True)


# write output to datastore under folder `outputdataset` and register it as a dataset after the experiment completes
# make sure the service principal in your datastore has blob data contributor role in order to write data back
datastore = workspace.get_default_datastore()
prepared_train_ds = OutputFileDatasetConfig(
    destination=(datastore, "outputdataset/{run-id}/train_data")
).register_on_complete(name="prepared_train_data")
prepared_val_ds = OutputFileDatasetConfig(
    destination=(datastore, "outputdataset/{run-id}/val_data")
).register_on_complete(name="prepared_val_data")


# Data Pre-processing

prep_env = Environment.from_conda_specification(
    name="prep-env", file_path="./environments/prep_conda_dependencies.yml"
)

prep_src = ScriptRunConfig(
    source_directory=script_folder,
    script="data_preprocessing.py",
    compute_target=compute_target,
    environment=prep_env,
)

prep_step = PythonScriptStep(
    name="prepare step",
    script_name=prep_src.script,
    # mount fashion_ds dataset to the compute_target
    arguments=[dataset.as_named_input("prepared_train_data"), prepared_train_ds],
    source_directory=prep_src.source_directory,
    runconfig=prep_src.run_config,
)


# Model Training

# (Neptune) When you use the same custom run ID for different run instances,
# you ensure that all metadata is logged to the same run.
NEPTUNE_CUSTOM_RUN_ID = hashlib.md5(str(time.time()).encode()).hexdigest()

train_env = Environment.from_conda_specification(
    name="train-env", file_path="./environments/train_conda_dependencies.yml"
)

train_src = ScriptRunConfig(
    source_directory=script_folder,
    script="train.py",
    compute_target=compute_target,
    environment=train_env,
)

train_step = PythonScriptStep(
    name="train step",
    script_name=train_src.script,
    arguments=[
        prepared_train_ds.read_delimited_files().as_input(name="prepared_train_data"),
        prepared_val_ds,
        "common/project-time-series-forecasting",
        NEPTUNE_CUSTOM_RUN_ID,
    ],
    source_directory=train_src.source_directory,
    runconfig=train_src.run_config,
)


# Model Validation

val_env = Environment.from_conda_specification(
    name="val-env", file_path="./environments/validate_conda_dependencies.yml"
)

val_src = ScriptRunConfig(
    source_directory=script_folder,
    script="validate.py",
    compute_target=compute_target,
    environment=val_env,
)

val_step = PythonScriptStep(
    name="validate step",
    script_name=val_src.script,
    arguments=[
        prepared_val_ds.read_delimited_files().as_input(name="prepared_val_data"),
        "common/project-time-series-forecasting",
        NEPTUNE_CUSTOM_RUN_ID,
    ],
    source_directory=val_src.source_directory,
    runconfig=val_src.run_config,
)


# Build Pipeline and run experiment
pipeline = Pipeline(workspace, steps=[prep_step, train_step, val_step])
run = exp.submit(pipeline)
run.wait_for_completion(show_output=True)
