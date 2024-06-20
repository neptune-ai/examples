import os
import shutil

import great_expectations as gx
import neptune
from great_expectations.data_context import EphemeralDataContext
from neptune.utils import stringify_unsupported

# Initialize a Neptune run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/great-expectations",
    tags=["script"],  # (optional) replace with your own
)

# Create a GX Data Context
context = gx.get_context()

## Upload context configuration to Neptune
run["gx/context/config"] = context.get_config().to_json_dict()

# Connect to Data
validator = context.sources.pandas_default.read_csv(
    "https://raw.githubusercontent.com/great-expectations/gx_tutorials/main/data/yellow_tripdata_sample_2019-01.csv"
)

# Create Expectations
validator.expect_column_values_to_not_be_null("pickup_datetime")
validator.expect_column_values_to_be_between("passenger_count", min_value=1, max_value=6)
validator.save_expectation_suite(
    discard_failed_expectations=False,
    discard_catch_exceptions_kwargs=False,
    discard_include_config_kwargs=False,
    discard_result_format_kwargs=False,
)

## Log Expectations to Neptune
expectation_suite = validator.get_expectation_suite().to_json_dict()

run["gx/meta"] = expectation_suite["meta"]

run["gx/expectations/expectations_suite_name"] = expectation_suite["expectation_suite_name"]

for idx, expectation in enumerate(expectation_suite["expectations"]):
    run["gx/expectations"][idx] = expectation

# Create Checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="my_quickstart_checkpoint",
    validator=validator,
)

## Log Checkpoint configuration to Neptune
run["gx/checkpoint/config"] = stringify_unsupported(checkpoint.config.to_json_dict())

# Run Validations
checkpoint_result = checkpoint.run()

## Log Validation results to Neptune
results_dict = checkpoint_result.list_validation_results()[0].to_json_dict()

run["gx/validations/json"] = results_dict

for idx, result in enumerate(results_dict["results"]):
    run["gx/validations/json/results"][idx] = result

# Upload HTML reports to Neptune
## Get the `local_site_path` of the Data Context

if isinstance(context, EphemeralDataContext):
    context = context.convert_to_file_context()

local_site_path = os.path.dirname(context.build_data_docs()["local_site"])[7:]

## Log Expectations reports to Neptune
run["gx/expectations/reports"].upload_files(os.path.join(local_site_path, "expectations"))

## Log Validations reports to Neptune
run["gx/validations/reports"].upload_files(os.path.join(local_site_path, "validations"))

# Cleanup
## Stop Neptune run
run.stop()

## (Optional) Delete the FileSystem Data Context
shutil.rmtree("gx")
