# Back up run metadata from Neptune

You can use this script to download run metadata from Neptune to your system. Note that this script is currently in beta, and we welcome your feedback and contributions to help improve it.

## Prerequisites

Before using this script, make sure you have
1. the Neptune environment variables set up. For instructions, see the [documentation](https://docs.neptune.ai/setup/setting_credentials/).
2. `tqdm` installed using `pip install -U tqdm`

## Instructions

The script will generate run logs in the same folder as `bulk_download_metadata.py`. You can modify this location in the `logging.basicConfig()` function.

To use the script, follow these steps:

1. Run `bulk_download_metadata.py`.
1. Enter the download path where you want the metadata to be downloaded.
1. Indicate if you want remotely tracked artifacts to be downloaded.
1. All eligible projects will be displayed in the console. Enter the projects you want to download the run metadata from.

## Download File types
The filetype of the downloaded metadata will depend on the Neptue field type to which it was logged.

Single values like parameters, `sys` and `monitoring` fields, etc. and `StringSet` like `sys/tags` will be logged to a `single_value_metadata.json` file under their respective `RUN_ID` folders. This JSON will have the flattened namespaces of the metadata as the keys.

| Neptune field fype | Downloaded file type
|:---:|:---:
|[Artifact](https://docs.neptune.ai/api/field_types/#artifact) / [File](https://docs.neptune.ai/api/field_types/#file) / [FileSeries](https://docs.neptune.ai/api/field_types/#fileseries)| Same as original
| [FloatSeries](https://docs.neptune.ai/api/field_types/#floatseries) / [StringSeries](https://docs.neptune.ai/api/field_types/#stringseries) | CSV
| [FileSet](https://docs.neptune.ai/api/field_types/#fileset) | ZIP
| Everything else | `single_value_metadata.json` |

## Download directory structure

All downloadable objects will follow the below folder structure inside the download directory:  
`WORKSPACE_NAME/PROJECT_NAME/RUN_ID/PATH/INSIDE/THE/RUN`


Example structure:
```
DOWNLOAD_FOLDER
├── WORKSPACE_1_FOLDER
│   ├── PROJECT_1_FOLDER
│   │   ├── RUN-1
│   │   │   ├── single_value_metadata.json
│   │   │   ├── metrics
│   │   │   │   ├── accuracy.csv
│   │   │   │   ├── loss.csv
│   │   │   │   ...
│   │   │   ├── monitoring
│   │   │   │   ├── <random hash 1>
│   │   │   │   │   ├── cpu.csv
│   │   │   │   │   ├── memory.csv
│   │   │   │   │   ...
│   │   │   │   ├── <random hash 2>
│   │   │   │   ...
│   │   │   ├── source_code
│   │   │   │   └── files.zip
│   │   │   ├── data
│   │   │   │   ├── train
│   │   │   │   │   ├── IMG1.jpg
│   │   │   │   │   ├── IMG2.jpg
│   │   │   │   │   ...
│   │   │   │   ├── test
│   │   │   │   ...
│   │   │   ...
│   │   ├── RUN-2
│   │   ...
│   ├── PROJECT_2_FOLDER
│   ...
├── WORKSPACE_2_FOLDER
...
```

## Note

There are a few things to keep in mind when using this script:

- Currently, only run metadata is downloaded. Project and model metadata are not downloaded†.
- All runs from the selected project will be downloaded. It is not currently possible to filter runs†.
- To download remotely tracked artifacts, the system you are running the script from must have access to the remote artifact location. Downloading remote artifacts can considerably increase execution times depending on the artifact sizes.

† Support for these can be added based on feedback

## Support

If you encounter any bugs or have feature requests, please submit them as [GitHub Issues](https://github.com/neptune-ai/examples/issues).

## License

Copyright (c) 2023, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
