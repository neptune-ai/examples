import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

CUSTOM_ENV_NAME = "neptune-example"
DEPENDENCIES_DIR = "./dependencies"

AZURE_SUBSCRIPTION_ID = "<YOUR SUBSCRIPTION ID>"
AZUREML_RESOURCE_GROUP_NAME = "<YOUR RESOURCE GROUP NAME>"
AZUREML_WORKSPACE_NAME = "<YOUR WORKSPACE NAME>"


def main():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential()

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZUREML_RESOURCE_GROUP_NAME,
        workspace_name=AZUREML_WORKSPACE_NAME,
    )

    pipeline_job_env = Environment(
        name=CUSTOM_ENV_NAME,
        description="Custom environment for Neptune Example",
        tags={"scikit-learn": "0.24.2"},
        conda_file=os.path.join(DEPENDENCIES_DIR, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        version="0.1.0",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    logging.info(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
    )


if __name__ == "__main__":
    main()
