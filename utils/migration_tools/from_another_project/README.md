# Exporting runs

This script allows you to export runs from one project to another within the same or different workspaces.

## Instructions

To use the script, follow these steps:

1. Execute `runs_migrator.py`.
2. Enter the source and target project names using the format `WORKSPACE_NAME/PROJECT_NAME`. A private target project will be created if it does not already exist.
3. Enter your API tokens from the source and target workspaces.
4. Enter the number of workers to use to copy the metadata. Leave blank to let `ThreadPoolExecutor` decide.
5. The script will generate run logs in the working directory. You can modify this location by editing the `logging.basicConfig()` function.
6. The source run of a migrated run can be idenfied from the `old_sys/run_id` field of the migrated run.

## Note

There are a few things to keep in mind when using this script:

- Avoid creating new runs in the source project while the script is running as these might not be copied.
- Currently, only run metadata is copied. Project and model metadata are not copied†.
- All runs from the source project will be copied to the target project. It is not possible to filter runs currently†.
- Most of the namespaces from the source runs will be retained in the target runs, except for the following:
  - `sys` namespace:
    - The `state` field cannot be copied.
    - The `description`, `name`, `custom_run_id`, `tags`, and `group_tags` fields are copied to the `sys` namespace in the target run.
    - All other fields are copied to a new `old_sys` namespace in the target run.
  - The `source_code/git` namespace cannot be copied.
- The relative time x-axis in copied charts is based on the `sys/creation_time` of the source runs. Since this field is read-only, the relative time will be negative in the copied charts, as the logging time occurred before the creation time of the target run.
- The hash of tracked artifacts may change between the source and target runs.
- File metadata is stored in the `.tmp_%Y%m%d%H%M%S` folder in the working directory. This folder can be deleted after the migration and sanity checks.

† Support for these can be added based on feedback

## Support

If you encounter any bugs or have feature requests, please submit them as [GitHub Issues](https://github.com/neptune-ai/examples/issues).

## License

Copyright (c) 2024, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
