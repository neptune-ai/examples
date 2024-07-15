# Migrating from the Model Registry to Experiments

This script allows you to copy models and model versions from the Model Registry to Experiments, within the same project.

#TODO: Mention benefits, Add screenshots

> [!NOTE]  FEEDBACK WELCOME
> This script is currently in beta, and we welcome your feedback and contributions to help improve it. Please submit any issues or feature requests as [GitHub Issues](https://github.com/neptune-ai/examples/issues)

## Prerequisites

Before using this script, make sure you have the Neptune environment variables set up. For instructions, see the [documentation](https://docs.neptune.ai/setup/setting_credentials/).

Additionally, ensure that the project you want to copy metadata to has already been created.

## Instructions

To use the script, follow these steps:

1. Execute `model_to_run.py`.
2. Enter the project name you want to copy the model metadata from, in the `WORKSPACE_NAME/PROJECT_NAME` format. Leave this prompt as black to use the `NEPTUNE_PROJECT` environment variable.
3. Enter the number of workers to use to copy the metadata. Leave blank to use all available CPUs.
4. The script will generate run logs in the same folder as `model_to_run.py`. You can modify this location by editing the `logging.basicConfig()` function.
5. The source object of a run can be identified using `sys/custom_run_id`.

## Note

There are a few things to keep in mind when using this script:

- All models and model versions will be copied. It is not possible to filter runs currently†.
- Most of the namespaces from the model/model_versions will be retained in the runs, except for the following:
  - `sys` namespace:
    - The `state` field cannot be copied.
    - The `description`, `name`, and `tags` fields are copied to the `sys` namespace in the new run.
    - All other fields are copied to a new `old_sys` namespace in the new run.
- File metadata is temporarily stored in a `tmp_tmp_%Y%m%d%H%M%S` folder in the current directory. This folder is deleted after the script finishes running.
- The Model ID of each model and corresponding model versions is stored as a group tag in the run to let you group models and model versions together.
- Runs made from models have the `model` tag, while those made from model versions have the `model_version` tag.
- The relative time x-axis in copied charts is based on the `sys/creation_time` of the original runs. Since this field is read-only, the relative time will be negative in the copied charts, as the logging time occurred before the creation time of the new run.
- The hash of tracked artifacts may change between the original and new runs.
- Each file copied as a `FileSet` will have its file name prefixed with the namespace where it was stored in the original run. For example, if the original run has a file named `hello_neptune.py` stored in the `source_code/files` namespace, the corresponding file in the new run will be named `source_code/files/hello_neptune.py`.

† Support for these can be added based on feedback

## Support

If you encounter any bugs or have feature requests, please submit them as [GitHub Issues](https://github.com/neptune-ai/examples/issues).

## License

Copyright (c) 2024, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
