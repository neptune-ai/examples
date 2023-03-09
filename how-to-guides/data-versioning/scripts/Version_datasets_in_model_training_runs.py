from pathlib import Path

import neptune
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# Download dataset
dataset_path = Path.relative_to(Path.absolute(Path(__file__)).parent, Path.cwd())

for file in ["train.csv", "test.csv"]:
    r = requests.get(
        f"https://raw.githubusercontent.com/neptune-ai/examples/main/how-to-guides/data-versioning/datasets/tables/{file}",
        allow_redirects=True,
    )

    open(dataset_path.joinpath(file), "wb").write(r.content)

TRAIN_DATASET_PATH = str(dataset_path.joinpath("train.csv"))
TEST_DATASET_PATH = str(dataset_path.joinpath("test.csv"))


def train_model(params, train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    FEATURE_COLUMNS = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    TARGET_COLUMN = ["variety"]
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    return rf.score(X_test, y_test)


#
# Run model training and log dataset version, parameter and test score to Neptune
#

# Create a Neptune run and start logging
run = neptune.init_run(project="common/data-versioning", api_token=neptune.ANONYMOUS_API_TOKEN)

# Track dataset version
run["datasets/train"].track_files(TRAIN_DATASET_PATH)
run["datasets/test"].track_files(TEST_DATASET_PATH)

# Log parameters
params = {
    "n_estimators": 3,
    "max_depth": 3,
    "max_features": 1,
}
run["parameters"] = params

# Calculate and log test score
score = train_model(params, TRAIN_DATASET_PATH, TEST_DATASET_PATH)
run["metrics/test_score"] = score

# Get the Neptune run ID of the first, baseline model training run
baseline_run_id = run["sys/id"].fetch()
print(baseline_run_id)

# Stop logging to the active Neptune run
run.stop()

#
# Run model training with different parameters and log metadata to Neptune
#

# Create a new Neptune run and start logging
new_run = neptune.init_run(project="common/data-versioning", api_token=neptune.ANONYMOUS_API_TOKEN)

# Track dataset version
new_run["datasets/train"].track_files(TRAIN_DATASET_PATH)
new_run["datasets/test"].track_files(TEST_DATASET_PATH)

# Query the baseline Neptune run
baseline_run = neptune.init_run(
    project="common/data-versioning",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    with_id=baseline_run_id,
    mode="read-only",
)

# Fetch the dataset version of the baseline model training run
baseline_run["datasets/train"].fetch_hash()

# Check if dataset versions changed or not between the runs
new_run.wait()  # force asynchronous logging operations to finish
assert baseline_run["datasets/train"].fetch_hash() == new_run["datasets/train"].fetch_hash()
assert baseline_run["datasets/test"].fetch_hash() == new_run["datasets/test"].fetch_hash()

# Select new parameters and log them to Neptune
params = {
    "n_estimators": 3,
    "max_depth": 2,
    "max_features": 3,
}
new_run["parameters"] = params

# Calculate the test score and log it to Neptune
score = train_model(params, TRAIN_DATASET_PATH, TEST_DATASET_PATH)
new_run["metrics/test_score"] = score

# Stop logging to the active Neptune run
new_run.stop()
baseline_run.stop()
#
# Go to Neptune to see how the results changed making sure that the training dataset versions were the same!
#
