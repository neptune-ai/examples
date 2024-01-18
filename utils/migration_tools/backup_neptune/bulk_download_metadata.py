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
# https://github.com/neptune-ai/examples/blob/main/utils/migration_tools/backup_neptune/README.md

VERSION = "0.1.1"


# %% Import libraries
import io
import json
import logging
import os
from contextlib import redirect_stdout
from datetime import datetime
from typing import Optional

import neptune
from neptune import management
from tqdm.auto import tqdm

_UNFETCHABLE_NAMESPACES = [
    "sys/state",
    "source_code/git",
]

_JSON_FILENAME = "simple_metadata.json"

# %% Set up logging
log_filename = datetime.now().strftime("neptune_backup_%Y%m%d%H%M%S.log")
print(f"Logs available at {log_filename}")

logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

logging.getLogger("neptune.internal.operation_processors.async_operation_processor").setLevel(
    logging.CRITICAL
)

logging.info("Backup process started")

# %%
download_folder = input(
    "Enter the download path (will be created if it doesn't exist). Leave blank to download to current directory: "
).strip()

if download_folder == "":
    download_folder = os.getcwd()

os.makedirs(download_folder, exist_ok=True)
logging.info(f"Downloading to {download_folder}")

# %%
download_artifacts = input("Download remotely tracked artifacts? (y|n):").strip() == "y"
logging.info(f"{download_artifacts=}")

# %%
projects = management.get_project_list()

print(f"Projects found: {projects}")
logging.info(f"Projects found: {projects}")

# %%
selected_projects = (
    input(
        "Enter projects you want to back up (comma-separated, no space) ('all' to export all projects): "
    )
    .strip()
    .lower()
)

logging.info(f"Exporting {selected_projects}")

if selected_projects == "all":
    selected_projects = projects
else:
    selected_projects = selected_projects.split(",")


# %%
def flatten_namespaces(
    dictionary: dict, prefix: Optional[list] = None, result: Optional[list] = None
) -> list:
    if prefix is None:
        prefix = []
    if result is None:
        result = []

    for k, v in dictionary.items():
        if isinstance(v, dict):
            flatten_namespaces(v, prefix + [k], result)
        elif prefix_str := "/".join(prefix):
            result.append(f"{prefix_str}/{k}")
        else:
            result.append(k)
    return result


# %% Start backup
print(f"Starting backup. View logs at {log_filename}. Press Ctrl/Cmd + C to cancel at any time")

for project in tqdm(selected_projects, desc="Total progress"):
    project_download_path = os.path.join(download_folder, project)
    os.makedirs(project_download_path, exist_ok=True)
    logging.info(f"Downloading runs from {project} to {project_download_path}")

    with redirect_stdout(io.StringIO()) as f:
        with neptune.init_project(project=project, mode="read-only") as _project:
            # Fetch runs table
            runs_table = _project.fetch_runs_table(columns=[]).to_pandas()

            if len(runs_table):
                runs = list(runs_table["sys/id"])

                for run_id in tqdm(runs, desc=project):
                    with neptune.init_run(
                        project=project,
                        with_id=run_id,
                        mode="read-only",
                    ) as run:
                        run_download_path = os.path.join(project_download_path, run_id)
                        os.makedirs(run_download_path, exist_ok=True)
                        logging.info(f"Downloading {project}/{run_id} to {run_download_path}")

                        namespaces = flatten_namespaces(run.get_structure())
                        single_values = {}

                        for namespace in namespaces:
                            if namespace in _UNFETCHABLE_NAMESPACES:
                                continue

                            namespace_download_path = os.path.join(run_download_path, namespace)

                            try:
                                if str(run[namespace]).split()[0] == "<Artifact":
                                    if download_artifacts:
                                        # Download artifact
                                        run[namespace].download(namespace_download_path)
                                elif str(run[namespace]).split()[0] == "<StringSet":
                                    # Write to single_values container
                                    single_values[namespace] = run[namespace].fetch()

                                elif str(run[namespace]).split()[0] in (
                                    "<FloatSeries",
                                    "<StringSeries",
                                ):
                                    # Download FloatSeries, StringSeries as CSV
                                    os.makedirs(
                                        os.path.dirname(namespace_download_path),
                                        exist_ok=True,
                                    )
                                    run[namespace].fetch_values().to_csv(
                                        f"{str(os.path.join(namespace_download_path))}.csv",
                                        index=False,
                                    )

                                elif str(run[namespace]).split()[0] == "<File":
                                    # Download File
                                    os.makedirs(
                                        os.path.dirname(namespace_download_path),
                                        exist_ok=True,
                                    )
                                    ext = run[namespace].fetch_extension()
                                    run[namespace].download(f"{namespace_download_path}.{ext}")

                                elif str(run[namespace]).split()[0] == "<FileSeries":
                                    # Download FileSeries
                                    run[namespace].download(namespace_download_path)

                                elif str(run[namespace]).split()[0] == "<FileSet":
                                    # Download FileSet
                                    os.makedirs(
                                        os.path.dirname(namespace_download_path),
                                        exist_ok=True,
                                    )
                                    run[namespace].download(f"{namespace_download_path}.zip")

                                else:
                                    # Write to single_values container
                                    single_values[namespace] = run[namespace].fetch()

                                # Export single_values container as json
                                with open(
                                    os.path.join(run_download_path, _JSON_FILENAME),
                                    mode="w+",
                                ) as file:
                                    file.write(
                                        json.dumps(
                                            single_values,
                                            indent=4,
                                            sort_keys=True,
                                            default=str,
                                        )
                                    )

                            except Exception as e:
                                logging.error(f"Error while downloading {namespace}\n{e}")
                                break
            else:
                logging.warning(f"No runs found in {project}")


logging.info("Backup complete!")

# %%
