# To fix the random RuntimeError: main thread is not in main loop error in Windows running python 3.8
import matplotlib.pyplot as plt
import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.utils import stringify_unsupported
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

plt.switch_backend("agg")

run = neptune.init_run(
    project="common/sklearn-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    name="other-options",
)

rfc = RandomForestClassifier()

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfc.fit(X_train, y_train)

# Log parameters from the model
run["estimator/parameters"] = stringify_unsupported(npt_utils.get_estimator_params(rfc))

# Log pickled model
run["estimator/pickled-model"] = npt_utils.get_pickled_model(rfc)

# Log confusion matrix
run["confusion-matrix"] = npt_utils.create_confusion_matrix_chart(
    rfc, X_train, X_test, y_train, y_test
)
