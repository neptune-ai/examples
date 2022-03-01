import neptune.new as neptune
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_DATASET_PATH = "../datasets/tables/train.csv"
TEST_DATASET_PATH = "../datasets/tables/test.csv"

PARAMS = {
    "n_estimators": 7,
    "max_depth": 2,
    "max_features": 2,
}


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

# Create Neptune Run and start logging
run = neptune.init(project="common/data-versioning", api_token="ANONYMOUS")

# Track dataset version
run["datasets/train"].track_files(TRAIN_DATASET_PATH)
run["datasets/test"].track_files(TEST_DATASET_PATH)

# Log parameters
run["parameters"] = PARAMS

# Calculate and log test score
score = train_model(PARAMS, TRAIN_DATASET_PATH, TEST_DATASET_PATH)
run["metrics/test_score"] = score

# Stop logging to the active Neptune Run
run.stop()

#
# Change the training data
# Run model training log dataset version, parameter and test score to Neptune
#

TRAIN_DATASET_PATH = "../datasets/tables/train_v2.csv"

# Create a new Neptune Run and start logging
new_run = neptune.init(project="common/data-versioning", api_token="ANONYMOUS")

# Log dataset versions
new_run["datasets/train"].track_files(TRAIN_DATASET_PATH)
new_run["datasets/test"].track_files(TEST_DATASET_PATH)

# Log parameters
new_run["parameters"] = PARAMS

# Calculate and log test score
score = train_model(PARAMS, TRAIN_DATASET_PATH, TEST_DATASET_PATH)
new_run["metrics/test_score"] = score

# Stop logging to the active Neptune Run
new_run.stop()

#
# Go to Neptune to see how the datasets changed between training runs!
#
