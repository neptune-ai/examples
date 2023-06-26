# Introduction
This script is intended to export runs from one project to another, in the same or different workspaces.  
**This is in beta and we invite feedback and contributions to improve this script.**

# Prerequisites
- A neptune.ai account, `neptune` python library installed, and environment variables set. Read the [docs](https://docs.neptune.ai/setup/installation/) to learn how to set up your installation.
- The project the runs need to be copied to should already be created.

# Instructions
- Run `runs_migrator.py`
- Enter the project names from and to which the runs will be copied, in the WORKSPACE_NAME/PROJECT_NAME format
- Run logs will be created in the same folder as `runs_migrator.py`. You can change this in `logging.basicConfig()`

# Caveats
- This currently only copies run metadata. Project and model metadata is not copied.
- Most of the namespaces will retained between the old and the new runs, except the below:
    - `sys` namespaces:
        - `state` cannot be copied
        - `description`, `name`, `custom_run_id`, and `tags` are copied to the `sys` namespace in the new run
        - All other namespaces are copied to a new `old_sys` namespace in the new run

    - `source_code/git` cannot be copied
- The relative time x-axis uses `sys/creation_time` as a reference. Since this field is read-only, relative time will be negative in copied charts as the logging time was before creation time of the new run
- The hash of tracked artifacts might change between the old and the new runs
- The file name of each file copied as a `FileSet` will be prefixed with the the namespace it was stored in the original run.  
For example, if the original run has the file `hello_neptune.py` stored in `source_code/files` namespace, the file name of the same file in the new run will be `source_code/files/hello_neptune.py`.

# Support
Submit bugs and feature requests as [GitHub Issues](https://github.com/neptune-ai/examples/issues).

# License

Copyright (c) 2022, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
