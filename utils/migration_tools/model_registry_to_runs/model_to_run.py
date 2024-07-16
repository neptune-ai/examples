#
# Copyright (c) 2024, Neptune Labs Sp. z o.o.
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
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from typing import Literal, Optional, Union

import neptune.metadata_containers
import pandas as pd
from neptune import management
from neptune.exceptions import MetadataInconsistency, MissingFieldException
from neptune.types import File
from tqdm.auto import tqdm

# %% Project Name
print(
    "Enter the project name (in WORKSPACE_NAME/PROJECT_NAME format). Leave empty to use the `NEPTUNE_PROJECT` environment variable`"
)

PROJECT = input().strip().lower() or os.getenv("NEPTUNE_PROJECT")

# %% Num Workers
print("Enter the number of workers to use (int). Leave empty to use all available CPUs")

NUM_WORKERS = int(input().strip() or os.cpu_count())

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

print(f"Logs available at {log_filename}\n")

# Silencing Neptune messages and urllib connection pool warnings
logging.getLogger("neptune").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# %% Create temporary directory to store local metadata
tmpdirname = "tmp_" + datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs(tmpdirname, exist_ok=True)
logging.info(f"Temporary directory created at {tmpdirname}")

# %% Map namespaces

READ_ONLY_NAMESPACES = {
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
    "sys/model_id",
}

MAPPED_NAMESPACES = {
    namespace: namespace.replace("sys", "old_sys") for namespace in READ_ONLY_NAMESPACES
}

UNFETCHABLE_NAMESPACES = {
    "sys/state",
}

# %% Validate project name
projects = management.get_project_list()

if PROJECT not in projects:
    logging.exception(f"Project {PROJECT} does not exist. Please check project name")
    exit()
else:
    logging.info(f"Copying Models in {PROJECT} to Runs using {NUM_WORKERS} workers")

# %% Get list of models to be copied
with neptune.init_project(
    project=PROJECT,
    mode="read-only",
) as neptune_from_project:
    models = neptune_from_project.fetch_models_table(columns=[]).to_pandas()["sys/id"].values

logging.info(f"{len(models)} models found")


# %% UDFs
@contextmanager
def threadsafe_change_directory(new_dir):
    lock = threading.Lock()
    old_dir = os.getcwd()
    try:
        with lock:
            os.chdir(new_dir)
        yield
    finally:
        with lock:
            os.chdir(old_dir)


def log_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Failed to copy {args[4]}/{args[1]} due to exception:\n{e}")

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
def copy_artifacts(object, namespace, run, id):
    for artifact_location in [
        artifact.metadata["location"] for artifact in object[namespace].fetch_files_list()
    ]:
        run[namespace].track_files(artifact_location)


@log_error
def copy_stringset(object, namespace, run, id):
    with contextlib.suppress(MissingFieldException, MetadataInconsistency):
        # Ignore missing `group_tags` field
        run[namespace].add(object[namespace].fetch())


@log_error
def copy_float_string_series(object, namespace, run, id):
    for row in object[namespace].fetch_values().itertuples():
        run[namespace].append(
            value=row.value,
            step=row.step,
            timestamp=time.mktime(pd.to_datetime(row.timestamp).timetuple()),
        )


@log_error
def copy_file(object, namespace, run, localpath, id):
    ext = object[namespace].fetch_extension()

    path = os.sep.join(namespace.split("/")[:-1])
    _download_path = os.path.join(localpath, path)
    os.makedirs(_download_path, exist_ok=True)
    object[namespace].download(_download_path, progress_bar=False)
    run[namespace].upload(os.path.join(localpath, namespace) + "." + ext)


@log_error
def copy_fileset(object, namespace, run, localpath, id):
    _download_path = os.path.join(localpath, namespace)
    os.makedirs(_download_path, exist_ok=True)
    object[namespace].download(_download_path, progress_bar=False)

    _zip_path = os.path.join(_download_path, f"{namespace.split('/')[-1]}.zip")
    with zipfile.ZipFile(_zip_path) as zip_ref:
        zip_ref.extractall(_download_path)
    os.remove(_zip_path)

    with threadsafe_change_directory(_download_path):
        run[namespace].upload_files(
            "*",
            wait=True,
        )


@log_error
def copy_fileseries(object, namespace, run, localpath, id):
    _download_path = os.path.join(localpath, namespace)
    object[namespace].download(_download_path, progress_bar=False)
    for file in glob(f"{_download_path}{os.sep}*"):
        run[namespace].append(File(file))


@log_error
def copy_atom(object, namespace, run, id):
    run[namespace] = object[namespace].fetch()


def copy_metadata(
    object: Union[neptune.Model, neptune.ModelVersion],
    object_id: str,
    run: neptune.Run,
) -> None:
    namespaces = flatten_namespaces(object.get_structure())

    _local_path = os.path.join(tmpdirname, object_id)

    for namespace in namespaces:
        if namespace in UNFETCHABLE_NAMESPACES:
            continue

        elif namespace in READ_ONLY_NAMESPACES:
            # Create old_sys namespaces for read-only sys namespaces
            run[MAPPED_NAMESPACES[namespace]] = object[namespace].fetch()

        elif str(object[namespace]).startswith("<Artifact"):
            copy_artifacts(object, namespace, run, object_id)

        elif str(object[namespace]).startswith("<StringSet"):
            copy_stringset(object, namespace, run, object_id)

        elif str(object[namespace]).split()[0] in (
            "<FloatSeries",
            "<StringSeries",
        ):
            copy_float_string_series(object, namespace, run, object_id)

        elif str(object[namespace]).startswith("<FileSet"):
            copy_fileset(object, namespace, run, _local_path, object_id)

        elif str(object[namespace]).startswith("<FileSeries"):
            copy_fileseries(object, namespace, run, _local_path, object_id)

        elif str(object[namespace]).startswith("<File"):
            copy_file(object, namespace, run, _local_path, object_id)

        else:
            copy_atom(object, namespace, run, object_id)

        run.wait()


def init_target_run(custom_run_id, type: Literal["model", "model_version"]):
    return neptune.init_run(
        project=PROJECT,
        custom_run_id=custom_run_id,  # Assigning model_id/model_version_id as custom_run_id to prevent duplication if the script is rerun
        tags=[type],
        capture_hardware_metrics=False,
        capture_stderr=False,
        capture_traceback=False,
        capture_stdout=False,
        git_ref=False,
        source_files=[],
    )


def copy_model_version(model_version_id, model_id):
    with neptune.init_model_version(
        project=PROJECT,
        with_id=model_version_id,
        mode="read-only",
    ) as model_version:
        with init_target_run(model_version_id, type="model_version") as model_version_run:
            model_version_run_id = model_version_run["sys/id"].fetch()

            # Adding model_id as a group tag for easier organization
            model_version_run["sys/group_tags"].add([model_id])

            copy_metadata(
                model_version,
                model_version_id,
                model_version_run,
            )

            logging.info(f"Copied {model_version_id} to {model_version_run_id}")


def copy_model(model_id):
    with neptune.init_model(
        project=PROJECT,
        with_id=model_id,
        mode="read-only",
    ) as model:
        with init_target_run(model_id, type="model") as model_run:
            model_run_id = model_run["sys/id"].fetch()

            # Adding model_id as a group tag for easier organization
            model_run["sys/group_tags"].add([model_id])

            copy_metadata(model, model_id, model_run)
            logging.info(f"Copied {model_id} to {model_run_id}")

            model_versions = model.fetch_model_versions_table(columns=[]).to_pandas()

            if model_versions.empty:
                return

            model_versions = (
                model.fetch_model_versions_table(columns=[]).to_pandas()["sys/id"].values
            )

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_model_version = {
                    executor.submit(
                        copy_model_version, model_version_id, model_id
                    ): model_version_id
                    for model_version_id in model_versions
                }

                for future in tqdm(
                    as_completed(future_to_model_version),
                    total=len(model_versions),
                    desc=f"Copying Model Versions for {model_id}",
                ):
                    model_version_id = future_to_model_version[future]
                    try:
                        future.result()
                    except Exception as e:
                        logging.exception(
                            f"Failed to copy {model_version_id} due to exception:\n{e}"
                        )


# %%
try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_model = {executor.submit(copy_model, model_id): model_id for model_id in models}

        for future in tqdm(
            as_completed(future_to_model),
            total=len(models),
            desc="Copying Models",
            position=0,
        ):
            model_id = future_to_model[future]
            try:
                future.result()
            except Exception as e:
                logging.exception(f"Failed to copy {model_id} due to exception:\n{e}")

        logging.info("Export complete!")

except Exception as e:
    logging.exception(f"Error during export: {e}")
    raise e

finally:
    logging.info(f"Cleaning up temporary directory {tmpdirname}")
    try:
        shutil.rmtree(tmpdirname)
        logging.info("Done!")
    except Exception as e:
        logging.exception(f"Failed to remove temporary directory {tmpdirname}\n{e}")
    finally:
        logging.shutdown()
