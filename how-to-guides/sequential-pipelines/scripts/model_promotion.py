import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.exceptions import NeptuneException
from neptune.new.types import File
from utils import *

model_name = "pickled_model"

# (Neptune) Get latest model from training stage
model_key = "PIPELINES"
project_key = "SHOW"

try:
    model = neptune.init_model(
        with_id=f"{project_key}-{model_key}",  # Your model ID here
    )
    model_versions_table = model.fetch_model_versions_table().to_pandas()
    staging_model_table = model_versions_table[model_versions_table["sys/stage"] == "staging"]
    challenger_model_id = staging_model_table ["sys/id"].tolist()[0]
    production_model_table = model_versions_table[model_versions_table["sys/stage"] == "production"]
    champion_model_id = production_model_table["sys/id"].tolist()[0]

except NeptuneException:
    print(
        f"The model with the provided key `{model_key}` doesn't exist in the `{project_key}` project."
    )

# (neptune) Download the lastest model checkpoint from model registry
challenger = neptune.init_model_version(with_id=challenger_model_id)
champion = neptune.init_model_version(with_id=champion_model_id)

# (Neptune) Get model weights from training stage
challenger[f"model/{model_name}"].download()
champion[f"model/{model_name}"].download()

# (Neptune) Move model to production
challenger_score = challenger["metrics/validation/scores/class_0"].fetch()
champion_score = champion["metrics/validation/scores/class_0"].fetch()

if challenger_score["fbeta_score"] > champion_score["fbeta_score"]:
    print(f"Promoting challenger model ID: {challenger_model_id}")
    challenger.change_stage("production")
    champion.change_stage("archived")
