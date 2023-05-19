#
# Copyright (c) 2023, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Instructions on how to use this script can be found at
# https://github.com/neptune-ai/examples/blob/main/utils/migration_tools/from_wandb/README.md

# %%
import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import neptune
import wandb
from neptune import management
from neptune.management.exceptions import ProjectNameCollision
from neptune.utils import stringify_unsupported
from tqdm.auto import tqdm

# %%
logging.basicConfig(
    filename=datetime.now().strftime("wandb_to_neptune_%Y%m%d%H%M%S.log"),
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

logging.getLogger("neptune.internal.operation_processors.async_operation_processor").setLevel(
    logging.CRITICAL
)

# %%
wandb_entity = input("Enter W&B entity name:").strip().lower()
neptune_workspace = input("Enter Neptune workspace name:").strip().lower()

logging.info(f"Exporting from W&B entity {wandb_entity} to Neptune workspace {neptune_workspace}")

# %%
client = wandb.Api()
wandb_projects = [project for project in client.projects()]
wandb_project_names = [project.name for project in wandb_projects]

logging.info(f"W&B projects found: {wandb_project_names}")
print(f"W&B projects found: {wandb_project_names}")

selected_projects = (
    input(
        "Enter projects you want to export (comma-separated, no space) ('all' to export all projects): "
    )
    .strip()
    .lower()
)

logging.info(f"Exporting {selected_projects}")

# %%
if selected_projects == "all":
    selected_projects = wandb_project_names
else:
    selected_projects = selected_projects.split(",")

# %%
for wandb_project in (
    project_pbar := tqdm(
        [project for project in wandb_projects if project.name in selected_projects]
    )
):
    project_pbar.set_description(f"Exporting {wandb_project.name}")
    wandb_project_name = wandb_project.name.replace("_", "-")

    # Create a new Neptune project for each W&B project
    # TODO: Let users chose if they want all W&B runs to be exported to a single Neptune project instead
    try:
        management.create_project(
            name=f"{neptune_workspace}/{wandb_project_name}",
            description="Exported from W&B",
        )
        logging.info(f"Created Neptune project {wandb_project_name}.")

    except ProjectNameCollision as e:
        logging.info(f"Project {wandb_project_name} already exists.")

    with neptune.init_project(
        project=f"{neptune_workspace}/{wandb_project_name}"
    ) as neptune_project:
        # Log W&B project URL to Neptune
        neptune_project["wandb_url"] = wandb_project.url

        wandb_runs = [run for run in client.runs(f"{wandb_entity}/{wandb_project.name}")]
        for wandb_run in (run_pbar := tqdm(wandb_runs)):
            run_pbar.set_description(f"Exporting {wandb_project_name}/{wandb_run.name}")
            # Initialize a new Neptune run for each W&B run
            with neptune.init_run(
                project=f"{neptune_workspace}/{wandb_project_name}",
                name=wandb_run.name,
                custom_run_id=wandb_run.id,
                description=wandb_run.notes,
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=False,
            ) as neptune_run:
                logging.info(f"Copying {wandb_run.url} to {neptune_run.get_url()}")

                # Fetch tags and parameters
                neptune_run["sys/tags"].add(wandb_run.tags)
                neptune_run["config"] = stringify_unsupported(wandb_run.config)

                # Add W&B metadata
                if wandb_run.job_type:
                    neptune_run["wandb/job_type"] = stringify_unsupported(wandb_run.job_type)
                neptune_run["wandb/path"] = "/".join(wandb_run.path)
                neptune_run["wandb/url"] = wandb_run.url
                neptune_run["wandb/created_at"] = wandb_run.created_at

                # Fetch summary
                # TODO: Create nested namespace structure
                summary = wandb_run.summary
                for key in summary.keys():
                    if key.startswith("_"):
                        continue
                    try:
                        if (
                            summary[key]["_type"] == "table-file"
                        ):  # Not uploading W&B table-file to Neptune
                            continue
                    except TypeError:
                        neptune_run["summary"][key] = stringify_unsupported(summary[key])
                    except KeyError:
                        continue

                # Fetch metrics
                history = wandb_run.history(stream="default")
                keys = [key for key in history.keys() if not key.startswith("_")]

                for i, record in enumerate(wandb_run.scan_history()):
                    step = record.get("_step")
                    epoch = record.get("epoch")
                    timestamp = record.get("_timestamp")
                    for key in keys:
                        value = record.get(key)
                        if value is None:
                            continue
                        try:
                            if (
                                dict(history[key])[i]["_type"] == "table-file"
                            ):  # Not uploading W&B table-file to Neptune
                                continue
                        except (IndexError, TypeError) as e:
                            try:
                                tag, name = key.rsplit("/", 1)
                                if "train" in tag:
                                    context = {"tag": tag, "subset": "train"}
                                elif "val" in tag:
                                    context = {"tag": tag, "subset": "val"}
                                elif "test" in tag:
                                    context = {"tag": tag, "subset": "test"}
                                else:
                                    context = {"tag": tag}
                            except ValueError:
                                name, context = key, {}
                            if context:
                                neptune_run[name] = context
                            elif timestamp:
                                neptune_run[name].append(value, timestamp=timestamp)
                            else:
                                neptune_run[name].append(value, step=step)

                # Add monitoring logs
                for record in wandb_run.history(stream="system", pandas=False):
                    timestamp = record.get("_timestamp")
                    for key in record:
                        if key.startswith("_"):  # Excluding '_runtime', '_timestamp', '_wandb'
                            continue

                        value = record.get(key)
                        if value is None:
                            continue

                        neptune_run["monitoring"][key.replace("system.", "")].append(
                            value, timestamp=timestamp
                        )

                # Fetch files
                excluded = {"artifact/", "config.yaml", "media/", "wandb-"}
                with TemporaryDirectory() as tmpdirname:
                    for file in wandb_run.files():
                        if (
                            file.size
                        ):  # A zero-byte file will be returned even when the `output.log` file does not exist
                            filename = file.name
                            if not any(filename.startswith(path) for path in excluded):
                                # Fetch console output logs
                                if filename == "output.log":
                                    try:
                                        file.download(root=tmpdirname)
                                        with open(Path(tmpdirname) / filename) as f:
                                            for line in f:
                                                neptune_run["monitoring/stdout"].append(line)
                                    except Exception as e:
                                        logging.warning(
                                            f"Failed to upload {filename} due to exception:\n{e}"
                                        )

                                # Fetch uploaded notebooks
                                elif filename == "code/_session_history.ipynb":
                                    try:
                                        file.download(root=tmpdirname)
                                        neptune_run["source_code/files"].upload_files(
                                            [f"{tmpdirname}/{filename}"]
                                        )
                                    except Exception as e:
                                        logging.warning(
                                            f"Failed to upload {filename} due to exception:\n{e}"
                                        )

                                # Fetch requirements.txt file
                                elif filename == "requirements.txt":
                                    try:
                                        file.download(root=tmpdirname)
                                        neptune_run["source_code/files"].upload_files(
                                            [f"{tmpdirname}/{filename}"]
                                        )
                                    except Exception as e:
                                        logging.warning(
                                            f"Failed to upload {filename} due to exception:\n{e}"
                                        )

                                # Fetch checkpoints
                                elif "ckpt" in file.name or "checkpoint" in file.name:
                                    try:
                                        file.download(root=tmpdirname)
                                        neptune_run["checkpoints"].upload_files(
                                            [f"{tmpdirname}/{filename}"]
                                        )
                                    except Exception as e:
                                        logging.warning(
                                            f"Failed to upload {filename} due to exception:\n{e}"
                                        )

                                # Fetch other files
                                else:
                                    try:
                                        file.download(root=tmpdirname)
                                        neptune_run["files"].upload_files(
                                            [f"{tmpdirname}/{filename}"]
                                        )
                                    except Exception as e:
                                        logging.warning(
                                            f"Failed to upload {filename} due to exception:\n{e}"
                                        )
                    neptune_run.wait()

# %%
logging.info("Export complete. Explore the metadata in the Neptune app.")
