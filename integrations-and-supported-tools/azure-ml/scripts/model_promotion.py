import logging
import os

import neptune
from neptune.exceptions import ModelNotFound

NEPTUNE_PROJECT = "common/project-time-series-forecasting"  # change to your own Neptune project

os.environ["NEPTUNE_PROJECT"] = "common/project-time-series-forecasting"


def promote_model():
    # (Neptune) Get latest model from training stage
    model_key = "PRO"
    project_key = "TSF"

    try:
        model = neptune.init_model(
            with_id=f"{project_key}-{model_key}",  # Your model ID here
        )
        model_versions_table = model.fetch_model_versions_table().to_pandas()
        staging_model_table = model_versions_table[model_versions_table["sys/stage"] == "staging"]
        challenger_model_id = staging_model_table["sys/id"].tolist()[0]
        production_model_table = model_versions_table[
            model_versions_table["sys/stage"] == "production"
        ]
        champion_model_id = production_model_table["sys/id"].tolist()[0]

    except ModelNotFound:
        logging.info(
            f"The model with the provided key `{model_key}` doesn't exist in the `{project_key}` project."
        )

    # (neptune) Download the lastest model checkpoint from model registry
    challenger = neptune.init_model_version(with_id=challenger_model_id)
    champion = neptune.init_model_version(with_id=champion_model_id)

    # (Neptune) Get model weights from training stage
    challenger["serialized_model"].download()
    champion["serialized_model"].download()

    # (Neptune) Move model to production
    challenger_score = challenger["scores"].fetch()
    champion_score = champion["scores"].fetch()

    logging.info(
        f"Challenger score: {challenger_score['rmse']}\nChampion score: {champion_score['rmse']}"
    )
    if challenger_score["rmse"] < champion_score["rmse"]:
        logging.info(
            f"Promoting challenger model {challenger_model_id} to production and archiving current champion model {champion_model_id}"
        )
        challenger.change_stage("production")
        champion.change_stage("archived")
    else:
        logging.info(
            f"Challenger model {challenger_model_id} underperforms champion {champion_model_id}. Archiving."
        )
        challenger.change_stage("archived")


if __name__ == "__main__":
    promote_model()
