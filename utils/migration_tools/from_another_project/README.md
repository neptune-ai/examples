# Exporting runs

This script allows you to export runs from one project to another within the same or different workspaces. Please note that this script is currently in beta, and we welcome your feedback and contributions to help improve it.

## Prerequisites

Before using this script, make sure you have the Neptune environment variables set up. For instructions, see the [documentation](https://docs.neptune.ai/setup/setting_credentials/).

Additionally, ensure that the project you want to copy runs to has already been created.

## Instructions

To use the script, follow these steps:

1. Execute `runs_migrator.py`.
2. Enter the project names from and to which you want to copy the runs, using the format `WORKSPACE_NAME/PROJECT_NAME`.
3. The script will generate run logs in the same folder as `runs_migrator.py`. You can modify this location by editing the `logging.basicConfig()` function.

## Note

There are a few things to keep in mind when using this script:

- Currently, only run metadata is copied. Project and model metadata are not copied†.
- All runs from the source project will be copied to the target project. It is not possible to filter runs currently†.
- Most of the namespaces from the original runs will be retained in the new runs, except for the following:
  - `sys` namespace:
    - The `state` field cannot be copied.
    - The `description`, `name`, `custom_run_id`, and `tags` fields are copied to the `sys` namespace in the new run.
    - All other fields are copied to a new `old_sys` namespace in the new run.
  - The `source_code/git` namespace cannot be copied.
- The relative time x-axis in copied charts is based on the `sys/creation_time` of the original runs. Since this field is read-only, the relative time will be negative in the copied charts, as the logging time occurred before the creation time of the new run.
- The hash of tracked artifacts may change between the original and new runs.
- Each file copied as a `FileSet` will have its file name prefixed with the namespace where it was stored in the original run. For example, if the original run has a file named `hello_neptune.py` stored in the `source_code/files` namespace, the corresponding file in the new run will be named `source_code/files/hello_neptune.py`.

† Support for these can be added based on feedback

## Support

If you encounter any bugs or have feature requests, please submit them as [GitHub Issues](https://github.com/neptune-ai/examples/issues).

## License

Copyright (c) 2023, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
