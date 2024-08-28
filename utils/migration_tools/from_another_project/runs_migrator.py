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
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from getpass import getpass
from glob import glob
from typing import Optional

import neptune.metadata_containers
import pandas as pd
from neptune import management
from neptune.exceptions import MetadataInconsistency, MissingFieldException
from neptune.types import File
from tqdm.auto import tqdm

# %% Project Name
SOURCE_PROJECT = input(
    "Enter the source project name (in WORKSPACE_NAME/PROJECT_NAME format): "
).strip()

TARGET_PROJECT = input(
    "Enter the target project name (in WORKSPACE_NAME/PROJECT_NAME format): "
).strip()

# %% API Tokens
SOURCE_TOKEN = getpass("Enter your API token for the source workspace: ")
TARGET_TOKEN = getpass("Enter your API token for the target workspace: ")

# %% Num Workers
NUM_WORKERS = input(
    "Enter the number of workers to use (int). Leave empty to use ThreadPoolExecutor's defaults: "
).strip()
NUM_WORKERS = None if NUM_WORKERS == "" else int(NUM_WORKERS)

# %% Setup logger
now = datetime.now()
log_filename = now.strftime(
    f"{SOURCE_PROJECT.replace('/','_')}_to_{TARGET_PROJECT.replace('/','_')}_%Y%m%d%H%M%S.log"
)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

logger = logging.getLogger(__name__)

print(f"Logs available at {log_filename}\n")


def exc_handler(exctype, value, tb):
    logger.exception("".join(traceback.format_exception(exctype, value, tb)))


sys.excepthook = exc_handler

# Silencing Neptune messages and urllib connection pool warnings
logging.getLogger("neptune").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# %% Create temporary directory to store local metadata
tmpdirname = os.path.abspath(os.path.join(os.getcwd(), ".tmp_" + now.strftime("%Y%m%d%H%M%S")))
os.mkdir(tmpdirname)
logger.info(f"Temporary directory created at {tmpdirname}")

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
}

MAPPED_NAMESPACES = {
    namespace: namespace.replace("sys", "old_sys") for namespace in READ_ONLY_NAMESPACES
}

UNFETCHABLE_NAMESPACES = {
    "sys/state",
    "sys/custom_run_id",  # This is being set separately
    "source_code/git",
}

# %% Validate project name

if SOURCE_PROJECT not in management.get_project_list(api_token=SOURCE_TOKEN):
    logger.error(f"Source project {SOURCE_PROJECT} does not exist. Please check project name")
    exit()

if TARGET_PROJECT not in management.get_project_list(api_token=TARGET_TOKEN):
    logger.info(f"Target project {TARGET_PROJECT} does not exist. Creating private project...")
    management.create_project(TARGET_PROJECT, api_token=TARGET_TOKEN)

logger.info(f"Copying runs from {SOURCE_PROJECT} to {TARGET_PROJECT}")

# %% Get list of runs to be copied
with neptune.init_project(
    project=SOURCE_PROJECT,
    api_token=SOURCE_TOKEN,
    mode="read-only",
) as neptune_from_project:
    source_runs = neptune_from_project.fetch_runs_table(columns=[]).to_pandas()["sys/id"].values

logger.info(f"{len(source_runs)} runs found")


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
            with contextlib.suppress(MissingFieldException):
                return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Failed to copy {args[-1]}/{args[1]} due to exception:\n{e}")

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
    for row in object[namespace].fetch_values(progress_bar=False).itertuples():
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
    object: neptune.Run,
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


def init_target_run(custom_run_id):
    return neptune.init_run(
        project=TARGET_PROJECT,
        api_token=TARGET_TOKEN,
        custom_run_id=custom_run_id,
        capture_hardware_metrics=False,
        capture_stderr=False,
        capture_traceback=False,
        capture_stdout=False,
        git_ref=False,
        source_files=[],
    )


def copy_run(source_run_id):
    with neptune.init_run(
        project=SOURCE_PROJECT,
        api_token=SOURCE_TOKEN,
        with_id=source_run_id,
        mode="read-only",
    ) as source_run:
        custom_run_id = None
        with contextlib.suppress(MissingFieldException):
            custom_run_id = source_run["sys/custom_run_id"].fetch()

        with init_target_run(custom_run_id) as target_run:
            target_run_id = target_run["sys/id"].fetch()

            copy_metadata(source_run, source_run_id, target_run)
            logger.info(
                f"Copied {SOURCE_PROJECT}/{source_run_id} to {TARGET_PROJECT}/{target_run_id}"
            )


# %%
try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_run = {
            executor.submit(copy_run, source_run_id): source_run_id for source_run_id in source_runs
        }

        for future in tqdm(as_completed(future_to_run), total=len(source_runs)):
            source_run_id = future_to_run[future]
            try:
                future.result()
            except Exception as e:
                logger.exception(f"Failed to copy {source_run_id} due to exception:\n{e}")

        logger.info("Export complete!")
        print("\nDone!")
except Exception as e:
    logger.exception(f"Error during export: {e}")
    print("\nError!")
    raise e

finally:
    logging.shutdown()
    print(f"Check logs at {log_filename}")
