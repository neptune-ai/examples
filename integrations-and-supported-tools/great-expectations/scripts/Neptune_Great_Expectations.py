import os
import shutil

import great_expectations as gx
import neptune
import pandas as pd
from great_expectations.data_context import EphemeralDataContext
from neptune.utils import stringify_unsupported

# Initialize a Neptune run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/great-expectations",
    tags=["script"],  # (optional) replace with your own
)

# Read data
df = pd.read_csv(
    "https://raw.githubusercontent.com/great-expectations/gx_tutorials/main/data/yellow_tripdata_sample_2019-01.csv"
)

# Create a GX Data Context
context = gx.get_context(mode="file")

# Upload context configuration to Neptune
run["gx/context/config"] = context.get_config().to_json_dict()

# Connect to Data
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

# Create Batch
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch-def")

batch_definition = (
    context.data_sources.get("pandas")
    .get_asset("pd dataframe asset")
    .get_batch_definition("batch-def")
)

batch_parameters = {"dataframe": df}

batch = batch_definition.get_batch(batch_parameters=batch_parameters)

# Create Expectation Suite
suite = gx.ExpectationSuite(name="expectation_suite")
suite = context.suites.add(suite)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="passenger_count", min_value=1, max_value=6
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(column="fare_amount", min_value=0)
)
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="pickup_datetime"))

# Log Expectations to Neptune
run["gx/meta"] = suite.meta

run["gx/expectations/expectations_suite_name"] = suite.name

for idx, expectation in enumerate(suite.to_json_dict()["expectations"]):
    run["gx/expectations"][idx] = expectation

# Create a Validation Definition
definition_name = "validation_definition"
validation_definition = gx.ValidationDefinition(
    data=batch_definition, suite=suite, name=definition_name
)

# Create Checkpoint
checkpoint_name = "my_checkpoint"

actions = [
    gx.checkpoint.UpdateDataDocsAction(name="update_all_data_docs"),
]

checkpoint = gx.Checkpoint(
    name=checkpoint_name,
    validation_definitions=[validation_definition],
    actions=actions,
    result_format={"result_format": "COMPLETE"},
)

context.validation_definitions.add(validation_definition)

context.checkpoints.add(checkpoint)

# Run Validations
results = checkpoint.run(batch_parameters=batch_parameters)

# Log Validation results to Neptune
run["gx/validations/success"] = results.describe_dict()["success"]

run["gx/validations/json"] = results.describe_dict()["validation_results"][0]

for idx, result in enumerate(results.describe_dict()["validation_results"][0]["expectations"]):
    run["gx/validations/json/results"][idx] = stringify_unsupported(result)

# Upload HTML reports to Neptune
if isinstance(context, EphemeralDataContext):
    context = context.convert_to_file_context()

local_site_path = os.path.dirname(context.build_data_docs()["local_site"])[7:]

# Upload Expectations Reports to Neptune
run["gx/expectations/reports"].upload_files(os.path.join(local_site_path, "expectations"))

# Upload Validations Reports to Neptune
run["gx/validations/reports"].upload_files(os.path.join(local_site_path, "validations"))

# Cleanup
# Stop Neptune run
run.stop()

# Delete FileSystem Data Context
shutil.rmtree("gx")
