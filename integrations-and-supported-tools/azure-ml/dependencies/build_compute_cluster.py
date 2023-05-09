import logging

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

AZURE_SUBSCRIPTION_ID = "<YOUR SUBSCRIPTION ID>"
AZUREML_RESOURCE_GROUP_NAME = "<YOUR RESOURCE GROUP NAME>"
AZUREML_WORKSPACE_NAME = "<YOUR WORKSPACE NAME>"


def create_compute_cluster() -> None:
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

    cpu_compute_target = "cpu-cluster"

    try:
        # let's see if the compute target already exists
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
        logging.info(
            f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
        )

    except Exception:
        logging.info("Creating a new cpu compute target...")

        # Let's create the Azure ML compute object with the intended parameters
        cpu_cluster = AmlCompute(
            # Name assigned to the compute cluster
            name="cpu-cluster",
            # Azure ML Compute is the on-demand VM service
            type="amlcompute",
            # VM Family
            size="STANDARD_NC6",
            # Minimum running nodes when there is no job running
            min_instances=0,
            # Nodes in cluster
            max_instances=4,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=180,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier="Dedicated",
        )

        # Now, we pass the object to MLClient's create_or_update method
        cpu_cluster = ml_client.begin_create_or_update(cpu_cluster)

    logging.info(
        f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}"
    )


if __name__ == "__main__":
    create_compute_cluster()
