import neptune
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite
from neptune.utils import stringify_unsupported
from sklearn import datasets

# Load sample data
iris_frame = datasets.load_iris(as_frame=True).frame

# Run Evidently test suites and reports
data_stability = TestSuite(
    tests=[
        DataStabilityTestPreset(),
    ]
)
data_stability.run(
    current_data=iris_frame.iloc[:60],
    reference_data=iris_frame.iloc[60:],
    column_mapping=None,
)

data_drift_report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)
data_drift_report.run(
    current_data=iris_frame.iloc[:60],
    reference_data=iris_frame.iloc[60:],
    column_mapping=None,
)

# (Neptune) Start a run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,  # replace with your own
    project="common/evidently-support",  # replace with your own
    tags=["reports"],  # (optional) replace with your own
)

# (Neptune) Save and upload reports as HTML
data_stability.save_html("data_stability.html")
data_drift_report.save_html("data_drift_report.html")

run["data_stability/report"].upload("data_stability.html")
run["data_drift/report"].upload("data_drift_report.html")

# (Neptune) Save reports as dict

run["data_stability"] = stringify_unsupported(data_stability.as_dict())
run["data_drift"] = stringify_unsupported(data_drift_report.as_dict())

# (Neptune) Stop logging
run.stop()
