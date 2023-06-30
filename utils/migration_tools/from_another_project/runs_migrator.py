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
# https://github.com/neptune-ai/examples/blob/main/utils/migration_tools/from_another_project/README.md


# %%
import contextlib
import logging
import os
import shutil
import time
from datetime import datetime
from glob import glob
from tempfile import TemporaryDirectory
from typing import Optional

import neptune
import pandas as pd
from neptune import management
from neptune.exceptions import (
    MissingFieldException,
    TypeDoesNotSupportAttributeException,
)
from neptune.types import File, GitRef
from tqdm.auto import tqdm

# %%
from_project = (
    input("Enter project name to migrate from in WORKSPACE_NAME/PROJECT_NAME format:")
    .strip()
    .lower()
)

# %%
to_project = (
    input("Enter project name to migrate to in WORKSPACE_NAME/PROJECT_NAME format:").strip().lower()
)

assert to_project != from_project, "To and from projects need to be different"
# %%

log_filename = datetime.now().strftime(
    f"{from_project.replace('/','_')}_to_{to_project.replace('/','_')}_%Y%m%d%H%M%S.log"
)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

print(f"Logs available at {log_filename}")

logging.getLogger("neptune.internal.operation_processors.async_operation_processor").setLevel(
    logging.CRITICAL
)

# %%
projects = management.get_project_list()

if from_project not in projects:
    logging.error(f"Project {from_project} does not exist. Please check project name")
elif to_project not in projects:
    logging.error(f"Project {to_project} does not exist. Please check project name")
else:
    logging.info(f"Copying from {from_project} to {to_project}")

# %% Get list of runs to be copied
with neptune.init_project(
    project=from_project,
    mode="read-only",
) as neptune_from_project:
    to_copy = neptune_from_project.fetch_runs_table(columns=[]).to_pandas()["sys/id"].values

logging.info(f"{len(to_copy)} runs found")


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


# %%

READ_ONLY_NAMESPACES = [
    "sys/creation_time",
    "sys/id",
    "sys/modification_time",
    "sys/monitoring_time",
    "sys/owner",
    "sys/ping_time",
    "sys/running_time",
    "sys/size",
    "sys/trashed",
]

MAPPED_NAMESPACES = {
    namespace: namespace.replace("sys", "old_sys") for namespace in READ_ONLY_NAMESPACES
}

UNFETCHABLE_NAMESPACES = [
    "sys/state",
    "sys/custom_run_id",  # This is being set separately
    "source_code/git",
]


# %%
for from_run_id in tqdm(to_copy):
    with neptune.init_run(
        project=from_project,
        with_id=from_run_id,
        mode="read-only",
    ) as from_run:
        custom_run_id = None

        with contextlib.suppress(MissingFieldException):
            custom_run_id = from_run["sys/custom_run_id"].fetch()

        with neptune.init_run(
            project=to_project,
            custom_run_id=custom_run_id or None,
            capture_hardware_metrics=False,
            capture_stderr=False,
            capture_traceback=False,
            git_ref=GitRef.DISABLED,
            source_files=[],
        ) as to_run:
            to_run_id = to_run["sys/id"].fetch()
            logging.info(f"Copying {from_project}/{from_run_id} to {to_project}/{to_run_id}")

            namespaces = flatten_namespaces(from_run.get_structure())

            for namespace in namespaces:
                if namespace in UNFETCHABLE_NAMESPACES:
                    continue

                elif namespace in READ_ONLY_NAMESPACES:
                    # Create old_sys namespaces for read-only sys namespaces
                    to_run[MAPPED_NAMESPACES[namespace]] = from_run[namespace].fetch()

                else:
                    try:
                        if str(from_run[namespace]).split()[0] == "<Artifact":
                            # Copy artifacts
                            for artifact_location in [
                                artifact.metadata["location"]
                                for artifact in from_run[namespace].fetch_files_list()
                            ]:
                                to_run[namespace].track_files(artifact_location)

                        elif str(from_run[namespace]).split()[0] == "<StringSet":
                            # Copy StringSet
                            to_run[namespace].add(from_run[namespace].fetch())

                        elif str(from_run[namespace]).split()[0] in (
                            "<FloatSeries",
                            "<StringSeries",
                        ):
                            # Copy FloatSeries, StringSeries
                            for row in from_run[namespace].fetch_values().itertuples():
                                to_run[namespace].append(
                                    value=row.value,
                                    step=row.step,
                                    timestamp=time.mktime(
                                        pd.to_datetime(row.timestamp).timetuple()
                                    ),
                                )
                        elif str(from_run[namespace]).split()[0] == "<File":
                            # Copy File
                            ext = from_run[namespace].fetch_extension()
                            with TemporaryDirectory(
                                suffix=f"_{to_run_id}",
                                prefix=f"{from_run_id}_",
                                dir=os.getcwd(),
                            ) as tmpdirname:
                                path = "/".join(namespace.split("/")[:-1])
                                os.makedirs(f"{tmpdirname}/{path}", exist_ok=True)
                                try:
                                    from_run[namespace].download(f"{tmpdirname}/{path}")
                                    to_run[namespace].upload(
                                        f"{tmpdirname}/{namespace}.{ext}",
                                        wait=True,
                                    )
                                except Exception as e:
                                    logging.error(
                                        f"Failed to copy {namespace}.{ext} due to exception:\n{e}"
                                    )
                        elif str(from_run[namespace]).split()[0] == "<FileSet":
                            # Copy FileSet
                            try:
                                os.makedirs(namespace, exist_ok=True)
                                from_run[namespace].download(namespace)
                                import zipfile

                                with zipfile.ZipFile(
                                    f"{namespace}/{namespace.split('/')[-1]}.zip", "r"
                                ) as zip_ref:
                                    zip_ref.extractall(namespace)
                                os.remove(f"{namespace}/{namespace.split('/')[-1]}.zip")
                                to_run[namespace].upload_files(
                                    namespace,
                                    wait=True,
                                )
                            except Exception as e:
                                logging.error(f"Failed to copy {namespace} due to exception:\n{e}")
                            else:
                                shutil.rmtree(namespace)
                        elif str(from_run[namespace]).split()[0] == "<FileSeries":
                            # Copy FileSeries
                            with TemporaryDirectory(
                                suffix=f"_{to_run_id}",
                                prefix=f"{from_run_id}_",
                                dir=os.getcwd(),
                            ) as tmpdirname:
                                try:
                                    from_run[namespace].download(tmpdirname)
                                    for file in glob(f"{tmpdirname}/*"):
                                        to_run[namespace].append(File(file), wait=True)
                                except Exception as e:
                                    logging.error(
                                        f"Failed to copy {namespace} due to exception:\n{e}"
                                    )
                        else:
                            to_run[namespace] = from_run[namespace].fetch()
                    except Exception as e:
                        logging.error(f"Error while copying {namespace}\n{e}")
                        break
            else:
                continue
            break

logging.info("Export complete!")
