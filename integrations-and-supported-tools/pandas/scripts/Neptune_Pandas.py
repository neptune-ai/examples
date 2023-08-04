from io import StringIO

import neptune
import pandas as pd
from neptune.types import File
from ydata_profiling import ProfileReport

# (Neptune) Initialize a run
run = neptune.init_run(
    project="common/pandas-support",
    api_token=neptune.ANONYMOUS_API_TOKEN,
)

# Load dataset
iris_df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    nrows=100,
)

# (Neptune) Log Dataframe as HTML
run["data/iris-df-html"].upload(File.as_html(iris_df))

# Save DataFrame as a CSV
csv_fname = "iris.csv"
iris_df.to_csv(csv_fname, index=False)

# (Neptune) Log CSV
run["data/iris-df-csv"].upload(csv_fname)

# Save DataFrame as a CSV buffer
csv_buffer = StringIO()
iris_df.to_csv(csv_buffer, index=False)

# (Neptune) Log CSV buffer
run["data/iris-df-csv-buffer"].upload(File.from_stream(csv_buffer, extension="csv"))

# More Options
# Log Pandas Profile Report to Neptune

# Create DataFrame profile report
profile = ProfileReport(iris_df, title="Iris Species Dataset Profile Report")

# (Neptune) Log Pandas profile report
run["data/iris-df-profile-report"].upload(File.from_content(profile.to_html(), extension="html"))
