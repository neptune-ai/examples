# Introduction
This script is intended to export run metadata from Weights and Biases to Neptune.  
**This is in beta and is offered without any warranties.**  
We invite contributions to improve this script.

# Prerequisites/assumptions
- A Weights and Biases account, `wandb` library installed, and environment variables set.
- A neptune.ai account, `neptune` python library installed, and environment variables set. Read the [docs](https://docs.neptune.ai/setup/installation/) to learn how to set up your installation.

# Instructions
- Run `wandb_to_neptune.py`
- Enter the W&B entity name from where you want to export runs, and the Neptune workspace name where you want to import experiments
- Enter the projects you want to export as comma-separated values, without spaces
- Run logs will be created in the same folder as `wandb_to_neptune.py`. This can be adjusted by updating the `logging.basicConfig()`

## Metadata mapping from W&B to Neptune

|     Metadata      |             W&B              |                 Neptune                  |
| :---------------: | :--------------------------: | :--------------------------------------: |
|   Project name    |       example_project        |       example-project<sup>1</sup>        |
|    Project URL    |         project.url          |            project.wandb_url             |
|     Run name      |           run.name           |               run.sys.name               |
|      Run ID       |            run.id            |    run.sys.custom_run_id<sup>2</sup>     |
|       Notes       |          run.notes           |           run.sys.description            |
|       Tags        |           run.tags           |               run.sys.tags               |
|      Config       |          run.config          |          run.config<sup>3</sup>          |
|     Job type      |         run.job_type         |            run.wandb.job_type            |
|     Run path      |           run.path           |              run.wandb.path              |
|      Run URL      |           run.url            |              run.wandb.url               |
|   Creation time   |        run.created_at        |           run.wandb.created_at           |
|    Run summary    |         run.summary          |         run.summary<sup>3</sup>          |
|    Run metrics    |      run.scan_history()      |      run.<METRIC_NAME><sup>4</sup>       |
|  System metrics   | run.history(stream="system") | run.monitoring.<METRIC_NAME><sup>5</sup> |
|    System logs    |          output.log          |          run.monitoring.stdout           |
|     Notebook      | code/_session_history.ipynb  |          run.source_code.files           |
| requirements.txt  |       requirements.txt       |          run.source_code.files           |
| Model checkpoints |    \*.ckpt/\*checkpoint\*    |             run.checkpoints              |
|    Other files    |         run.files()          |                run.files                 |

<sup>1</sup>Underscores `_` in a W&B project name are replaced by a hyphen `-` in Neptune  
<sup>2</sup>Passing the wandb.run.id as neptune.run.custom_run_id ensures that duplicate Neptune runs are not created for the same W&B run even if the script is run multiple times  
<sup>2</sup>Values are converted to a string in Neptune  
<sup>3</sup>`_steps` and `_timestamps` associated with a metric are logged as `step` and `timestamp` respectively with a Neptune metric  
<sup>4</sup>`system.` prefix is removed when logging to Neptune

# What is not exported
- Models
- W&B table-file†
- `run.summary` keys starting with `_`†
- Metrics starting with `_`†
- Files with path starting with any of `artifact/`, `config.yaml`, `media/`, `wandb-`†

†These have been excluded at the code level to prevent redundancy and noise, but can be included.

# Potential issues and workarounds