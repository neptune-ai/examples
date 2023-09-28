# Backup run metadata from Neptune

This script allows you to download run metadata from Neptune to your system. Please note that this script is currently in beta, and we welcome your feedback and contributions to help improve it.

## Prerequisites

Before using this script, make sure you have the Neptune environment variables set up. For instructions, see the [documentation](https://docs.neptune.ai/setup/setting_credentials/).

## Instructions

To use the script, follow these steps:

1. Run `bulk_download_metadata.py`.
1. The script will generate run logs in the same folder as `bulk_download_metadata.py`. You can modify this location by editing the `logging.basicConfig()` function.
1. Enter the download path where you want the metadata to be downloaded.
1. Indicate if you want remotely tracked artifacts to be downloaded.
1. Enter the projects you want to download the run metadata from.

## Download directory structure
```
DOWNLOAD_FOLDER
|---WORKSPACE_1_FOLDER
    |---PROJECT_1_FOLDER
    |---PROJECT_2_FOLDER
    ...
|---WORKSPACE_2_FOLDER
...
```

## Note

There are a few things to keep in mind when using this script:

- Currently, only run metadata is downloaded. Project and model metadata are not downloaded†.
- All runs from the selected project will be downloaded. It is not possible to filter runs currently†.
- Downloading remotely tracked artifacts will require that the system you are running the script from has access to the remote artifact location, and can considerably increase execution times depending on the artifact sizes.

† Support for these can be added based on feedback

## Support

If you encounter any bugs or have feature requests, please submit them as [GitHub Issues](https://github.com/neptune-ai/examples/issues).

## License

Copyright (c) 2023, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
