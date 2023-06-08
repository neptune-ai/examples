# Azure Machine Learning how-to

This project is an example integration between Azure DevOps and Azure ML services with Neptune.

## Environment preparation

### Azure ML

To run the example, first create a compute cluster and a custom environment in your Azure ML environment. You can do that by executing the `./dependencies/build_compute_cluster.py` and `./dependencies/build_environment.py` scripts.

Note that you will need to fill

```
AZURE_SUBSCRIPTION_ID = "<YOUR SUBSCRIPTION ID>"
AZUREML_RESOURCE_GROUP_NAME = "<YOUR RESOURCE GROUP NAME>"
AZUREML_WORKSPACE_NAME = "<YOUR WORKSPACE NAME>"
```
with values representing your environment.

### Azure DevOps

For Azure DevOps Pipelines to be able to successfully create and execute Azure ML Pipelines, create the following secrets as per `./azure-ci/azure-pipelines.yaml` in your Azure DevOps Pipeline via the UI:

```
AZURE_TENANT_ID: $(tenant)
AZURE_CLIENT_ID: $(client)
AZURE_CLIENT_SECRET: $(secret)
NEPTUNE_API_TOKEN: $(neptune-sa-token)
```

## The example
The example is focused around creation of an Azure DevOps CI/CD pipeline that would be able to test the Azure ML Pipeline and then deploy it for operational purposes. The following picture shows the resulting AzureML pipeline:

<p align="center">
  <img src="https://neptune.ai/wp-content/uploads/2023/04/Screenshot-2023-04-17-at-16.32.20.png" height="500"/>
</p>
