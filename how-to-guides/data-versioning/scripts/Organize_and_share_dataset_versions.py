from pathlib import Path

import neptune
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# Download dataset
dataset_path = Path.relative_to(Path.absolute(Path(__file__)).parent, Path.cwd())

for file in ["train_sampled.csv", "test.csv"]:
    r = requests.get(
        f"https://raw.githubusercontent.com/neptune-ai/examples/main/how-to-guides/data-versioning/datasets/tables/{file}",
        allow_redirects=True,
    )

    open(dataset_path.joinpath(file), "wb").write(r.content)

# Initialize Neptune project
project = neptune.init_project(
    project="common/data-versioning", api_token=neptune.ANONYMOUS_API_TOKEN
)

# Create a few versions of a dataset and save them to Neptune
train = pd.read_csv(str(dataset_path.joinpath("train.csv")))

for i in range(5):
    train_sample = train.sample(frac=0.5 + 0.1 * i)
    train_sample.to_csv(str(dataset_path.joinpath("train_sampled.csv")), index=None)
    project[f"datasets/train_sampled/v{i}"].track_files(
        str(dataset_path.joinpath("train_sampled.csv")), wait=True
    )

print(project.get_structure())


# Get the latest version of the dataset and save it as 'latest'


def get_latest_version():
    artifact_name = project.get_structure()["datasets"]["train_sampled"].keys()
    versions = [int(version.replace("v", "")) for version in artifact_name if version != "latest"]
    return max(versions)


latest_version = get_latest_version()
print("latest version", latest_version)

project["datasets/train_sampled/latest"].assign(
    project[f"datasets/train_sampled/v{latest_version}"].fetch(), wait=True
)

print(project.get_structure()["datasets"])

# Create a Neptune run
run = neptune.init_run(project="common/data-versioning", api_token=neptune.ANONYMOUS_API_TOKEN)

# Assert that you are training on the latest dataset
TRAIN_DATASET_PATH = str(dataset_path.joinpath("train_sampled.csv"))
run["datasets/train"].track_files(TRAIN_DATASET_PATH, wait=True)

assert run["datasets/train"].fetch_hash() == project["datasets/train_sampled/latest"].fetch_hash()

TEST_DATASET_PATH = str(dataset_path.joinpath("test.csv"))

# Log parameters
params = {
    "n_estimators": 8,
    "max_depth": 3,
    "max_features": 2,
}
run["parameters"] = params

# Train the model
train = pd.read_csv(TRAIN_DATASET_PATH)
test = pd.read_csv(TEST_DATASET_PATH)

FEATURE_COLUMNS = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
TARGET_COLUMN = ["variety"]
X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)

# Save the score
score = rf.score(X_test, y_test)
run["metrics/test_score"] = score

#
# Go to the Neptune app to see datasets logged at the Project level!
#
