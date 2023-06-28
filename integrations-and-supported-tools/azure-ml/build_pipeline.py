import os
import time
import uuid

from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

DATA_PREP_SRC_DIR = "components/data_prep"
TRAIN_SRC_DIR = "components/train"
VALID_SRC_DIR = "components/validate"

NEPTUNE_PROJECT = "common/project-time-series-forecasting"  # change to your own Neptune project
NEPTUNE_CUSTOM_RUN_ID = str(uuid.uuid4())
NEPTUNE_API_TOKEN = os.environ["NEPTUNE_API_TOKEN"]

AZURE_SUBSCRIPTION_ID = "<YOUR SUBSCRIPTION ID>"
AZUREML_RESOURCE_GROUP_NAME = "<YOUR RESOURCE GROUP NAME>"
AZUREML_WORKSPACE_NAME = "<YOUR WORKSPACE NAME>"


def compose_pipeline(
    compute_target="cpu-cluster",
    custom_env_name="neptune-example",
    custom_env_version="2",
    neptune_project=NEPTUNE_PROJECT,
    neptune_custom_run_id="",
):
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        credential = InteractiveBrowserCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZUREML_RESOURCE_GROUP_NAME,
        workspace_name=AZUREML_WORKSPACE_NAME,
    )

    web_path = "https://raw.githubusercontent.com/neptune-ai/examples/main/use-cases/time-series-forecasting/walmart-sales/dataset/aggregate_data.csv"

    aggregate_data = Data(
        name="aggregate_data",
        path=web_path,
        type=AssetTypes.URI_FILE,
        description="Dataset for credit card defaults",
        tags={"source_type": "web", "source": "UCI ML Repo"},
        version="1.0.0",
    )

    aggregate_data = ml_client.data.create_or_update(aggregate_data)

    data_prep_component = command(
        name="data_prep",
        display_name="Data preparation for training",
        description="reads a .csv input, prepares it for training",
        inputs={"data": Input(type="uri_folder")},
        outputs=dict(train_data=Output(type="uri_folder", mode="rw_mount")),
        code=DATA_PREP_SRC_DIR,
        command="""python data_preprocessing.py \
                --data ${{inputs.data}} \
                --train_data ${{outputs.train_data}}
                """,
        environment=f"{custom_env_name}:{custom_env_version}",
    )

    train_component = command(
        name="train",
        display_name="Model training",
        description="reads a .csv input, splits into training and validation, trains model and outputs validation dataset",
        inputs={
            "train_data": Input(type="uri_folder"),
            "neptune_project": neptune_project,
            "neptune_custom_run_id": neptune_custom_run_id,
            "neptune_api_token": NEPTUNE_API_TOKEN,
        },
        outputs=dict(valid_data=Output(type="uri_folder", mode="rw_mount")),
        code=TRAIN_SRC_DIR,
        command="""python train.py \
                --train_data ${{inputs.train_data}} \
                --valid_data ${{outputs.valid_data}} \
                --neptune_project ${{inputs.neptune_project}} \
                --neptune_custom_run_id ${{inputs.neptune_custom_run_id}} \
                --neptune_api_token ${{inputs.neptune_api_token}}
                """,
        environment=f"{custom_env_name}:{custom_env_version}",
    )

    valid_component = command(
        name="train",
        display_name="Model validation",
        description="reads a .csv input and validates it against a validation dataset",
        inputs={
            "valid_data": Input(type="uri_folder"),
            "neptune_project": neptune_project,
            "neptune_custom_run_id": neptune_custom_run_id,
            "neptune_api_token": NEPTUNE_API_TOKEN,
        },
        code=VALID_SRC_DIR,
        command="""python validate.py \
                --valid_data ${{inputs.valid_data}} \
                --neptune_project ${{inputs.neptune_project}} \
                --neptune_custom_run_id ${{inputs.neptune_custom_run_id}} \
                --neptune_api_token ${{inputs.neptune_api_token}}
                """,
        environment=f"{custom_env_name}:{custom_env_version}",
    )

    @dsl.pipeline(
        compute=compute_target,
        description="E2E neptune-example pipeline",
    )
    def ml_pipeline(
        pipeline_job_data_input,
    ):
        data_prep_job = data_prep_component(data=pipeline_job_data_input)

        train_job = train_component(
            train_data=data_prep_job.outputs.train_data,
        )

        valid_job = valid_component(
            valid_data=train_job.outputs.valid_data,
        )

    pipeline = ml_pipeline(pipeline_job_data_input=Input(type="uri_file", path=aggregate_data.path))

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name="neptune_example",
    )

    ml_client.jobs.stream(pipeline_job.name)


if __name__ == "__main__":
    compose_pipeline(neptune_project=NEPTUNE_PROJECT, neptune_custom_run_id=NEPTUNE_CUSTOM_RUN_ID)
