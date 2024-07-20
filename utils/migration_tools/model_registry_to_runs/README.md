# Migrating from the Model Registry to Experiments

This script allows you to copy models and model versions from the model registry to experiments, within the same project.

## Why Migrate?
Our revamped web app adds a host of new features to experiments, including better grouping and enhanced reporting capabilities. By migrating your models and model versions to experiments, you can take advantage of these features to better organize and compare your models.

As a part of our focus on performance this year, we have also highly optimized the `Run` API  for speed. Using the `Run` object instead of `Model` or `ModelVersion` objects will allow you to take advantage of these optimizations.

This script leverages the new *Group Tags* to organize your models and model versions in experiments, and *tags* to create saved views separating model metadata runs from training runs. As a result, you can view all models and model versions in the same table:

![Models and model versions displayed in the Experiments tab in the Neptune app](https://neptune.ai/wp-content/uploads/2024/07/MRtoRun.png)

In this screenshot, _Group Tag_, _Custom Run ID_, and _ID_ correspond to the original Model ID, Model Version ID and the new Run ID respectively.

Having all model metadata in experiments also lets you use experiment's native [comparisons](https://docs.neptune.ai/usage/tutorial/#compare-the-runs) and [reports](https://docs.neptune.ai/app/reports/) to compare models and model versions.


## Prerequisites

Before using this script, make sure you have the Neptune environment variables set up. For instructions, see the [documentation](https://docs.neptune.ai/setup/setting_credentials/).

## Instructions

To use the script, follow these steps:

1. Execute `model_to_run.py`.
2. Enter the name of a project from which you want to copy the model metadata. Use the `WORKSPACE_NAME/PROJECT_NAME` format. To use the `NEPTUNE_PROJECT` environment variable, leave this prompt blank.
3. Enter the number of workers to use to copy the metadata. Leave blank to let `ThreadPoolExecutor` decide.
4. The script will generate run logs in the working directory. You can modify this location by editing the `logging.basicConfig()` function.


## Note

There are a few things to keep in mind when using this script:

- Avoid creating new models/model versions while the script is running as these might not be copied
- All models and model versions will be copied. Filtering is currently not available†.
- Most of the namespaces from the model/model_versions will be retained in the runs, except for the following:
  - `sys` namespace:
    - The `state` field cannot be copied.
    - The `description`, `name`, and `tags` fields are copied to the `sys` namespace in the new run.
    - All other fields are copied to a new `old_sys` namespace in the new run.
- The _Model Stage_ is currently copied to `old_sys/stage` field. Unlike the `sys/stage` field, this field cannot be updated from the web app. If you want to be able to update the _Model Stage_ from the web app, the script can be modified to copy the stage as _Tags_ instead†.
- File metadata is stored in the `tmp_%Y%m%d%H%M%S` folder in the working directory.
- The relative time x-axis in copied charts is based on the `sys/creation_time` of the original runs. Since this field is read-only, the relative time will be negative in the copied charts, as the logging time occurred before the creation time of the new run.
- The hash of tracked artifacts may change between the original and new runs.
- Each file copied as a `FileSet` will have its file name prefixed with the namespace where it was stored in the original run. For example, if the original run has a file named `hello_neptune.py` stored in the `source_code/files` namespace, the corresponding file in the new run will be named `source_code/files/hello_neptune.py`.

† Can be added based on feedback

## Post-Migration
- The source object of a run can be identified using `sys/custom_run_id`.
- The Model ID of each model and corresponding model versions is stored as a group tag in the run to allow you to group models and model versions together, as shown in the screenshot above.
- Runs made from models and model versions have the the `model` and `model_version` tags added respectively.
- Once the migration and any sanity checks are complete, the copied Model/Model Versions and the temporary directory can be deleted from the model registry and working directory respectively to reclaim space.
- This script can also be used a template to update your logging script to start logging model metadata to runs instead of the model registry. For example, `init_model()` and `init_model_version()` calls will need to be replaced by `init_run(tags=["model"])` and `init_run(tags=["model-version"])` respectively.

## Performance Benchmarks

The script was tested on a project with 86 models and model versions, totaling 510MB spread across metrics and files using `neptune==1.10.4`.  

On an internet connection with download and upload speeds of 340Mbps and 110Mbps, respectively, and an average round-trip time to the Neptune server of 28ms, the entire migration took ~60 seconds when using 20 workers and ~6 minutes when using only 1.

## Roadmap

- [ ] Copy models and model versions from multiple or all projects of the workspace
- [ ] Filter models and model versions to copy
- [ ] Copy the _Model Stage_ as a tag instead of a field

## Support and Feedback

We welcome your feedback and contributions to help improve the script. Please submit any issues or feature requests as [GitHub Issues](https://github.com/neptune-ai/examples/issues)

## License

Copyright (c) 2024, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
