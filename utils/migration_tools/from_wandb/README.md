# Migrating from W&B to Neptune

This script allows you to copy run metadata from W&B to Neptune.

## Prerequisites
- A Weights and Biases account, `wandb` library installed, and environment variables set.
- A neptune.ai account, `neptune` python library installed, and environment variables set. Read the [docs](https://docs.neptune.ai/setup/installation/) to learn how to set up your installation.

## Instructions

To use the script, follow these steps:

1. Run `wandb_to_neptune.py`
2. Enter the source W&B entity name. Leave blank to use your default entity
3 Enter the destination Neptune workspace name. Leave blank to read from the `NEPTUNE_PROJECT` environment variable
3. Enter the number of workers to use to copy the metadata. Leave blank to use all available CPUs
4. Enter the W&B projects you want to export as comma-separated values. Enter *all* to export all projects
5. The script will generate run logs in the working directory. You can change this in `logging.basicConfig()`. Live progress bars will also be rendered in the console.
6. Private Neptune projects corresponding to the W&B projects will be created if they do not exist. The project description will be set as *Exported from <W&B project URL>*. This can be changed later.

## Metadata mapping from W&B to Neptune

| Metadata | W&B | Neptune |
| :-: | :-: | :-: |
| Project name | example_project | example-project<sup>1</sup> |
| Project URL | project.url | project.wandb_url |
| Run name | run.name | run.sys.name |
| Run ID | run.id | run.sys.custom_run_id<sup>2</sup> |
| Notes | run.notes | run.sys.description |
| Tags | run.tags | run.sys.tags |
| Group | run.group | run.sys.group_tags |
| Config | run.config | run.config<sup>3</sup> |
| Run summary | run.summary | run.summary<sup>3</sup> |
| Run metrics | run.scan_history() | run.<METRIC_NAME><sup>4</sup> |
| System metrics | run.history(stream="system") | run.monitoring.<METRIC_NAME><sup>5</sup> |
| System logs | output.log | run.monitoring.stdout |
| Source Code | code/* | run.source_code.files |
| requirements.txt | requirements.txt | run.source_code.requirements |
| Model checkpoints | \*.ckpt/\*checkpoint\* | run.checkpoints |
| Other files | run.files() | run.files |
| All W&B attributes | run.* | run.wandb.* |

<sup>1</sup> Underscores `_` in a W&B project name are replaced by a hyphen `-` in Neptune  
<sup>2</sup> Passing the wandb.run.id as neptune.run.custom_run_id ensures that duplicate Neptune runs are not created for the same W&B run even if the script is run multiple times  
<sup>3</sup> Values are converted to a string in Neptune  
<sup>4</sup> `_steps` and `_timestamps` associated with a metric are logged as `step` and `timestamp` respectively with a Neptune metric  
<sup>5</sup> `system.` prefix is removed when logging to Neptune

## What is not exported
- Models
- W&B specific objects and datatypes
- `run.summary` keys starting with `_`†
- Metrics and W&B attributes starting with `_`†
- Files with path starting with any of `artifact/`, `config.yaml`, `media/`, `wandb-`†

† These have been excluded at the code level to prevent redundancy and noise, but can be included.

## Post-Migration
* W&B Workspace views can be recreated using Neptune's [overlaid charts](https://docs.neptune.ai/app/charts/) and [reports](https://docs.neptune.ai/app/reports/)
* W&B Runs table views can be recreated using Neptune's [Custom Views](https://docs.neptune.ai/app/experiments/#custom-views)
  ![Example W&B Runs table view recreated in Neptune](https://neptune.ai/wp-content/uploads/2024/07/wandb_table.png)
* W&B Run Overview can be recreated using Neptune's [Custom Dashboards](https://docs.neptune.ai/app/custom_dashboard/)
    ![Example W&B Run Overview recreated in Neptune](https://neptune.ai/wp-content/uploads/2024/07/overview.png)

## Performance Benchmarks

The script was tested on by copying 32 W&B runs across 8 projects, totaling ~1MB spread across metrics and files.  
On an internet connection with download and upload speeds of 340Mbps and 110Mbps, respectively, and an average round-trip time of 18ms and 19ms respectively to Neptune and W&B servers, the entire process took ~30 seconds using 20 workers.

Neptune client version 1.10.4 and wandb client version 0.17.4 were used.

## Support and Feedback

We welcome your feedback and contributions to help improve the script. Please submit any issues or feature requests as [GitHub Issues](https://github.com/neptune-ai/examples/issues)

## License

Copyright (c) 2024, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
