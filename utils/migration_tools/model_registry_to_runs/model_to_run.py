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
import functools
import logging
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from glob import glob
from typing import Optional, Union

import neptune.metadata_containers
import pandas as pd
from neptune import management
from neptune.exceptions import MetadataInconsistency, MissingFieldException
from neptune.types import File
from tqdm.auto import tqdm

# %%
print("Enter the project name (in WORKSPACE_NAME/PROJECT_NAME format)")
print("Leave blank to use the `NEPTUNE_PROJECT` environment variable`")

PROJECT = input().strip().lower() or os.getenv("NEPTUNE_PROJECT")

# %% Setup logger

log_filename = datetime.now().strftime(
    f"models_migration_{PROJECT.replace('/','_')}_%Y%m%d%H%M%S.log"
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

# Silencing Neptune messages
logging.getLogger("neptune").setLevel(logging.CRITICAL)

# %% Create temporary directory to store local metadata
tmpdirname = "tmp_" + datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs(tmpdirname, exist_ok=True)
logging.info(f"Temporary directory created at {tmpdirname}")

# %% Map namespaces

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
    "sys/stage",
]

MAPPED_NAMESPACES = {
    namespace: namespace.replace("sys", "old_sys") for namespace in READ_ONLY_NAMESPACES
}

UNFETCHABLE_NAMESPACES = [
    "sys/state",
]

# %% Validate project name
projects = management.get_project_list()

if PROJECT not in projects:
    logging.error(f"Project {PROJECT} does not exist. Please check project name")
    exit()
else:
    logging.info(f"Copying Models in {PROJECT} to Runs")

# %% Get list of models to be copied
with neptune.init_project(
    project=PROJECT,
    mode="read-only",
) as neptune_from_project:
    models = neptune_from_project.fetch_models_table(columns=[]).to_pandas()["sys/id"].values

logging.info(f"{len(models)} models found")


# %% UDFs


def log_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Failed to copy {args[1]} due to exception:\n{e}")

    return wrapper


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


@log_error
def copy_artifacts(object, namespace, run):
    for artifact_location in [
        artifact.metadata["location"] for artifact in object[namespace].fetch_files_list()
    ]:
        run[namespace].track_files(artifact_location)


@log_error
def copy_stringset(object, namespace, run):
    with contextlib.suppress(MissingFieldException, MetadataInconsistency):
        # Ignore missing `group_tags` field
        run[namespace].add(object[namespace].fetch())


@log_error
def copy_float_string_series(object, namespace, run):
    for row in object[namespace].fetch_values().itertuples():
        run[namespace].append(
            value=row.value,
            step=row.step,
            timestamp=time.mktime(pd.to_datetime(row.timestamp).timetuple()),
        )


@log_error
def copy_file(object, namespace, run, localpath):
    ext = object[namespace].fetch_extension()

    path = os.pathsep.join(namespace.split("/")[:-1])
    _download_path = os.path.join(localpath, path)
    os.makedirs(_download_path, exist_ok=True)
    object[namespace].download(_download_path, progress_bar=False)
    run[namespace].upload(os.path.join(localpath, namespace) + "." + ext)


@log_error
def copy_fileset(object, namespace, run, localpath):
    _download_path = os.path.join(localpath, namespace)
    os.makedirs(_download_path, exist_ok=True)
    object[namespace].download(_download_path, progress_bar=False)

    _zip_path = os.path.join(_download_path, f"{namespace.split('/')[-1]}.zip")
    with zipfile.ZipFile(_zip_path) as zip_ref:
        zip_ref.extractall(_download_path)
    os.remove(_zip_path)
    run[namespace].upload_files(
        _download_path,
    )


@log_error
def copy_fileseries(object, namespace, run, localpath):
    _download_path = os.path.join(localpath, namespace)
    object[namespace].download(_download_path, progress_bar=False)
    for file in glob(f"{tmpdirname}{os.pathsep}*"):
        run[namespace].append(File(file))


@log_error
def copy_atom(object, namespace, run):
    run[namespace] = object[namespace].fetch()


def copy_metadata(
    object: Union[neptune.Model, neptune.ModelVersion],
    id: str,
    run: neptune.Run,
) -> None:
    """
    Copy metadata from a Neptune Model or ModelVersion to a Run.

    Args:
        object: The Neptune Model or ModelVersion to copy metadata from.
        id: The ID of the object.
        run: The Neptune Run to copy metadata to.

    Returns:
        None
    """

    namespaces = flatten_namespaces(object.get_structure())

    _local_path = os.path.join(tmpdirname, id)

    for namespace in namespaces:
        if namespace in UNFETCHABLE_NAMESPACES:
            continue

        elif namespace in READ_ONLY_NAMESPACES:
            # Create old_sys namespaces for read-only sys namespaces
            run[MAPPED_NAMESPACES[namespace]] = object[namespace].fetch()

        else:
            try:
                if str(object[namespace]).startswith("<Artifact"):
                    copy_artifacts(object, namespace, run)

                elif str(object[namespace]).startswith("<StringSet"):
                    copy_stringset(object, namespace, run)

                elif str(object[namespace]).split()[0] in (
                    "<FloatSeries",
                    "<StringSeries",
                ):
                    copy_float_string_series(object, namespace, run)

                elif str(object[namespace]).startswith("<File"):
                    copy_file(object, namespace, run, _local_path)

                elif str(object[namespace]).startswith("<FileSet"):
                    copy_fileset(object, namespace, run, _local_path)

                elif str(object[namespace]).startswith("<FileSeries"):
                    copy_fileseries(object, namespace, run, _local_path)

                else:
                    copy_atom(object, namespace, run)

            except Exception as e:
                logging.error(f"Error while copying {id}\{namespace}\n{e}")
                break
            else:
                continue


def copy_model_version(model_version_id):
    with neptune.init_model_version(
        project=PROJECT,
        with_id=model_version_id,
        mode="read-only",
    ) as model_version:
        with init_target_run(model_version_id) as model_version_run:
            model_version_run_id = model_version_run["sys/id"].fetch()

            logging.info(f"Copying {model_version_id} to {model_version_run_id}")

            # Adding model_id as a group tag for easier organization
            model_version_run["sys/group_tags"].add([model_id])

            copy_metadata(
                model_version,
                model_version_id,
                model_version_run,
            )


def init_target_run(custom_run_id):
    return neptune.init_run(
        project=PROJECT,
        custom_run_id=custom_run_id,  # Assigning model_id/model_version_id as custom_run_id to prevent duplication if the script is rerun
        tags=["model"],
        capture_hardware_metrics=False,
        capture_stderr=False,
        capture_traceback=False,
        capture_stdout=False,
        git_ref=False,
        source_files=[],
    )


def copy_model(model_id):
    with neptune.init_model(
        project=PROJECT,
        with_id=model_id,
        mode="read-only",
    ) as model:
        with init_target_run(model_id) as model_run:
            model_run_id = model_run["sys/id"].fetch()

            logging.info(f"Copying {model_id} to {model_run_id}")

            # Adding model_id as a group tag for easier organization
            model_run["sys/group_tags"].add([model_id])

            copy_metadata(model, model_id, model_run)

            model_versions = model.fetch_model_versions_table(columns=[]).to_pandas()

            if model_versions.empty:
                logging.info(f"0 model versions found within {model_id}")
                return

            model_versions = (
                model.fetch_model_versions_table(columns=[]).to_pandas()["sys/id"].values
            )

            logging.info(f"{len(model_versions)} model_versions found within {model_id}")

            model_version_pbar = tqdm(model_versions, position=1, leave=False)

            for model_version_id in model_version_pbar:
                model_version_pbar.set_description(f"Copying {model_version_id} metadata")
                copy_model_version(model_version_id)


# %%
try:
    model_pbar = tqdm(models, position=0)

    for model_id in model_pbar:
        model_pbar.set_description(f"Copying {model_id} metadata")
        copy_model(model_id)

    logging.info("Export complete!")

except Exception as e:
    logging.error(f"Error during export: {e}")
    raise e

finally:
    logging.info(f"Cleaning up temporary directory {tmpdirname}")
    try:
        shutil.rmtree(tmpdirname)
        logging.info("Done!")
    except Exception as e:
        logging.error(f"Failed to remove temporary directory {tmpdirname}\n{e}")
    finally:
        logging.shutdown()
