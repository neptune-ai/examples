import logging
import os

import neptune
from neptune.exceptions import ModelNotFound

NEPTUNE_PROJECT = "common/project-time-series-forecasting"  # change to your own Neptune project

os.environ["NEPTUNE_PROJECT"] = NEPTUNE_PROJECT


def download_latest_prod_model():
    model_key = "PRO"
    project_key = "TSF"

    try:
        model = neptune.init_model(
            with_id=f"{project_key}-{model_key}",  # Your model ID here
        )
        model_versions_table = model.fetch_model_versions_table().to_pandas()
        production_model_table = model_versions_table[
            model_versions_table["sys/stage"] == "production"
        ]
        prod_model_id = production_model_table["sys/id"].tolist()[0]

    except ModelNotFound:
        logging.info(
            f"The model with the provided key `{model_key}` doesn't exist in the `{project_key}` project."
        )

    # (neptune) Download the lastest model checkpoint from model registry
    prod_model = neptune.init_model_version(with_id=prod_model_id)

    # (Neptune) Get model weights from training stage
    prod_model["serialized_model"].download()

    logging.info(
        f"Model to be deployed: {prod_model_id}. Model has been downloaded and is ready for deployment."
    )


if __name__ == "__main__":
    download_latest_prod_model()
