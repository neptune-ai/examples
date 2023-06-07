import warnings

import dalex as dx
import neptune
import pandas as pd
from neptune.types import File
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Load data
data = dx.datasets.load_titanic()

X = data.drop(columns="survived")
y = data.survived

# Create a pipeline model
numerical_features = ["age", "fare", "sibsp", "parch"]
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["gender", "class", "embarked"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=500, random_state=0)

clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

# Fit the model
clf.fit(X, y)

# (Neptune) Start a run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,  # replace with your own
    project="common/dalex-support",  # replace with your own
    tags=["dalex reports"],  # (optional) replace with your own
)

# Create an explainer for the model
exp = dx.Explainer(clf, X, y)

# (Neptune) Upload explainer object to Neptune
run["pickled_explainer"].upload(File.from_content(exp.dumps()))

# Model-level explanations
mp = exp.model_performance()
vi = exp.model_parts()
vi_grouped = exp.model_parts(
    variable_groups={
        "personal": ["gender", "age", "sibsp", "parch"],
        "wealth": ["class", "fare"],
    }
)
pdp_num = exp.model_profile(type="partial", label="pdp")
ale_num = exp.model_profile(type="accumulated", label="ale")
pdp_cat = exp.model_profile(
    type="partial",
    variable_type="categorical",
    variables=["gender", "class"],
    label="pdp",
)
ale_cat = exp.model_profile(
    type="accumulated",
    variable_type="categorical",
    variables=["gender", "class"],
    label="ale",
)

# (Neptune) Upload model-level explanation plots to Neptune
run["model/performance/roc"].upload(mp.plot(geom="roc", show=False))
run["model/variable_importance/single"].upload(vi.plot(show=False))
run["model/variable_importance/grouped"].upload(vi_grouped.plot(show=False))
run["model/profile/num"].upload(pdp_num.plot(ale_num, show=False))
run["model/profile/cat"].upload(ale_cat.plot(pdp_cat, show=False))

# Prediction-level explanations
## Create sample data
john = pd.DataFrame(
    {
        "gender": ["male"],
        "age": [25],
        "class": ["1st"],
        "embarked": ["Southampton"],
        "fare": [72],
        "sibsp": [0],
        "parch": 0,
    },
    index=["John"],
)

mary = pd.DataFrame(
    {
        "gender": ["female"],
        "age": [35],
        "class": ["3rd"],
        "embarked": ["Cherbourg"],
        "fare": [25],
        "sibsp": [0],
        "parch": [0],
    },
    index=["Mary"],
)

# Create explanations on sample predictions
bd_john = exp.predict_parts(john, type="break_down", label=john.index[0])
bd_interactions_john = exp.predict_parts(john, type="break_down_interactions", label="John+")
sh_mary = exp.predict_parts(mary, type="shap", B=10, label=mary.index[0])
cp_mary = exp.predict_profile(mary, label=mary.index[0])
cp_john = exp.predict_profile(john, label=john.index[0])

# (Neptune) Upload prediction-level explanation plots to Neptune
run["prediction/breakdown/john"].upload(bd_john.plot(bd_interactions_john, show=False))
run["prediction/shapely/mary"].upload(sh_mary.plot(show=False))
run["prediction/profile/numerical"].upload(cp_mary.plot(cp_john, show=False))
run["prediction/profile/categorical"].upload(
    cp_mary.plot(cp_john, variable_type="categorical", show=False)
)

# (Neptune) Stop logging
run.stop()
