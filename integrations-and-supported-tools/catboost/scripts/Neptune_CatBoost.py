# Neptune + CatBoost

## Import dependencies
import neptune
from catboost import CatBoostClassifier
from catboost.datasets import titanic
from neptune.types import File
from neptune.utils import stringify_unsupported
from sklearn.model_selection import train_test_split

## (Neptune) Start a run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,  # Replace with your own
    project="common/catboost-support",  # Replace with your own
    tags=["catboost", "classifier", "script"],  # (optional) use your own
)

## Load data
titanic_train, titanic_test = titanic()

### (Neptune) Upload raw data
run["data/raw/train"].upload(File.as_html(titanic_train))
run["data/raw/test"].upload(File.as_html(titanic_test))

### Preprocess data
titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace=True)
titanic_train["Cabin"].fillna("", inplace=True)
titanic_train["Embarked"].fillna(titanic_train["Embarked"].mode()[0], inplace=True)

titanic_test["Age"].fillna(titanic_test["Age"].median(), inplace=True)
titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)
titanic_test["Cabin"].fillna("", inplace=True)

label = ["Survived"]
cat_features = ["Sex", "Embarked"]
text_features = ["Name", "Ticket", "Cabin"]

X_train, X_eval, y_train, y_eval = train_test_split(
    titanic_train.drop(columns=label + ["PassengerId"]),
    titanic_train[label],
    test_size=0.25,
    shuffle=True,
)

### (Neptune) Upload processed data
run["data/processed/train"].upload(File.as_html(titanic_train))
run["data/processed/test"].upload(File.as_html(titanic_test))

## Train a CatBoost model

model = CatBoostClassifier()

plot_file = "training_plot.html"

model.fit(
    X=X_train,
    y=y_train,
    eval_set=(X_eval, y_eval),
    cat_features=cat_features,
    text_features=text_features,
    plot_file=plot_file,
    use_best_model=True,
)

### (Neptune) Upload training results
#### Upload training plot
run["training/plot"].upload(plot_file)

#### Upload training metrics

run["training/best_score"] = stringify_unsupported(model.get_best_score())
run["training/best_iteration"] = stringify_unsupported(model.get_best_iteration())

## Make predictions
titanic_test["prediction"] = model.predict(
    data=titanic_test.drop(columns=["PassengerId"]),
    prediction_type="Class",
)

### (Neptune) Upload predictions
titanic_test.to_csv("results.csv", index=False)

run["data/results"].upload("results.csv")

## (Neptune) Upload model metadata to Neptune
### Upload model binary
model.save_model("model.cbm")

run["model/binary"].upload("model.cbm")

### Upload model attributes
run["model/attributes/tree_count"] = model.tree_count_
run["model/attributes/feature_importances"] = dict(
    zip(model.feature_names_, model.get_feature_importance())
)
run["model/attributes/probability_threshold"] = model.get_probability_threshold()

### Upload model parameters
run["model/parameters"] = stringify_unsupported(model.get_all_params())

## Stop logging
run.stop()

## Analyze run in the Neptune app
# Follow the run link in the console output and explore the logged metadata.
# You can also explore this example run
# https://app.neptune.ai/o/common/org/catboost-support/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=Overview-99f571df-0fec-4447-9ffe-5a4c668577cd&shortId=CAT-2
