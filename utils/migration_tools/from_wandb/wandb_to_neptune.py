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
# https://github.com/neptune-ai/examples/blob/main/utils/migration_tools/from_wandb/README.md
# %%
import functools
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from datetime import datetime

import neptune
import wandb
from neptune import management
from neptune.management.exceptions import ProjectNameCollision
from neptune.utils import stringify_unsupported
from tqdm.auto import tqdm

# %%
wandb.require("core")
client = wandb.Api(timeout=120)

# %% Input prompts
if client.default_entity:
    wandb_entity = (
        input(
            f"Enter W&B entity name. Leave blank to use the default entity ({client.default_entity}): "
        ).strip()
        or client.default_entity
    )
else:
    wandb_entity = input("Enter W&B entity name: ").strip()

if default_neptune_workspace := os.getenv("NEPTUNE_PROJECT"):
    default_neptune_workspace = default_neptune_workspace.split("/")[0]
    neptune_workspace = (
        input(
            f"Enter Neptune workspace name. Leave blank to use the default workspace ({default_neptune_workspace}): "
        ).strip()
        or default_neptune_workspace
    )
else:
    neptune_workspace = input("Enter Neptune workspace name: ").strip()

num_workers = input(
    "Enter the number of workers to use (int). Leave empty to use ThreadPoolExecutor's defaults: "
).strip()

num_workers = None if num_workers == "" else int(num_workers)

# %% Setup logging
now = datetime.now()
log_filename = now.strftime("wandb_to_neptune_%Y%m%d%H%M%S.log")

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

# Silencing Neptune messages, W&B errors, and urllib connection pool warnings
logging.getLogger("neptune").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("wandb").setLevel(logging.ERROR)

logger.info(f"Copying from W&B entity {wandb_entity} to Neptune workspace {neptune_workspace}")

# %% Create temporary directory to store local metadata
tmpdirname = os.path.abspath(os.path.join(os.getcwd(), "tmp_" + now.strftime("%Y%m%d%H%M%S")))
os.makedirs(tmpdirname, exist_ok=True)
logger.info(f"Temporary directory created at {tmpdirname}")

# %%
wandb_projects = [project for project in client.projects()]  # sourcery skip: identity-comprehension
wandb_project_names = [project.name for project in wandb_projects]

print(f"W&B projects found ({len(wandb_project_names)}): {wandb_project_names}")

selected_projects = input(
    "Enter projects you want to copy (comma-separated). Leave blank to copy all projects:  "
).strip()

# %%
if selected_projects == "":
    selected_projects = wandb_project_names
else:
    selected_projects = [project.strip() for project in selected_projects.split(",")]

if not_found := set(selected_projects) - set(wandb_project_names):
    print(f"Projects not found: {not_found}")
    logger.warning(f"Projects not found: {not_found}")
    selected_projects = set(selected_projects) - not_found

print(f"Copying {len(selected_projects)} projects: {selected_projects}\n")
logger.info(f"Copying {len(selected_projects)} projects: {selected_projects}")


# %% UDFs
def log_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to copy {args[1]} due to exception:\n{e}")

    return wrapper


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


def copy_run(wandb_run: client.run, wandb_project_name: str) -> None:
    with neptune.init_run(
        project=f"{neptune_workspace}/{wandb_project_name}",
        name=wandb_run.name,
        custom_run_id=wandb_run.id,
        description=wandb_run.notes,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False,
        capture_traceback=False,
        source_files=[],
        git_ref=False,
        tags=wandb_run.tags,
    ) as neptune_run:
        # Add W&B run attributes

        for attr in wandb_run._attrs:
            try:
                if (
                    attr.startswith("user")
                    or attr.startswith("_")
                    or callable(getattr(wandb_run, attr))
                ):
                    continue
                if attr == "group":
                    neptune_run["sys/group_tags"].add(wandb_run.group)
                elif attr == "config":
                    neptune_run["config"] = stringify_unsupported(wandb_run.config)
                else:
                    neptune_run[f"wandb/{attr}"] = stringify_unsupported(getattr(wandb_run, attr))

            except TypeError:
                pass
            except Exception as e:
                logger.error(f"Failed to copy {wandb_run.attr} due to exception:\n{e}")

        copy_summary(neptune_run, wandb_run)
        copy_metrics(neptune_run, wandb_run)
        copy_monitoring_metrics(neptune_run, wandb_run)
        copy_files(neptune_run, wandb_run)
        neptune_run.wait()
        logger.info(f"Copied {wandb_run.url} to {neptune_run.get_url()}")


def copy_summary(neptune_run: neptune.Run, wandb_run: client.run) -> None:
    summary = wandb_run.summary
    for key in summary.keys():
        if key.startswith("_"):
            continue
        try:
            if summary[key]["_type"] == "table-file":  # Not uploading W&B table-file to Neptune
                continue
        except TypeError:
            neptune_run["summary"][key] = stringify_unsupported(summary[key])
        except KeyError:
            continue


def copy_metrics(neptune_run: neptune.Run, wandb_run: client.run) -> None:
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
            except (IndexError, TypeError):
                if epoch:
                    neptune_run[key].append(value, step=epoch)
                elif timestamp:
                    neptune_run[key].append(value, timestamp=timestamp)
                else:
                    neptune_run[key].append(value, step=step)


def copy_monitoring_metrics(neptune_run: neptune.Run, wandb_run: client.run) -> None:
    for record in wandb_run.history(stream="system", pandas=False):
        timestamp = record.get("_timestamp")
        for key in record:
            if key.startswith("_"):  # Excluding '_runtime', '_timestamp', '_wandb'
                continue

            value = record.get(key)
            if value is None:
                continue

            neptune_run["monitoring"][key.replace("system.", "")].append(value, timestamp=timestamp)


@log_error
def copy_console_output(neptune_run: neptune.Run, download_path: str) -> None:
    with open(download_path) as f:
        for line in f:
            neptune_run["monitoring/stdout"].append(line)


@log_error
def copy_source_code(
    neptune_run: neptune.Run,
    download_path: str,
    filename: str,
) -> None:
    with threadsafe_change_directory(os.path.join(download_path.replace(filename, ""), "code")):
        neptune_run["source_code/files"].upload_files(filename.replace("code/", ""), wait=True)


@log_error
def copy_requirements(neptune_run: neptune.Run, download_path: str) -> None:
    neptune_run["source_code/requirements"].upload(download_path)


@log_error
def copy_other_files(
    neptune_run: neptune.Run, download_path: str, filename: str, namespace: str
) -> None:
    with threadsafe_change_directory(download_path.replace(filename, "")):
        neptune_run[namespace].upload_files(filename, wait=True)


def copy_files(neptune_run: neptune.Run, wandb_run: client.run) -> None:
    EXCLUDED_PATHS = {"artifact/", "config.yaml", "media/", "wandb-"}
    download_folder = os.path.join(tmpdirname, wandb_run.project, wandb_run.id)
    for file in wandb_run.files():
        if file.size and not any(
            file.name.startswith(path) for path in EXCLUDED_PATHS
        ):  # A zero-byte file will be returned even when the `output.log` file does not exist
            download_path = os.path.join(download_folder, file.name)
            try:
                file.download(root=download_folder, replace=True, exist_ok=True)
                if file.name == "output.log":
                    copy_console_output(neptune_run, download_path)

                elif file.name.startswith("code/"):
                    copy_source_code(neptune_run, download_path, file.name)

                elif file.name == "requirements.txt":
                    copy_requirements(neptune_run, download_path)

                elif "ckpt" in file.name or "checkpoint" in file.name:
                    copy_other_files(neptune_run, download_path, file.name, namespace="checkpoints")

                else:
                    copy_other_files(neptune_run, download_path, file.name, namespace="files")
            except Exception as e:
                logger.error(f"Failed to copy {download_path} due to exception:\n{e}")


def copy_project(wandb_project: client.project) -> None:
    # sourcery skip: identity-comprehension
    wandb_project_name = wandb_project.name.replace("_", "-")

    # Create a new Neptune project for each W&B project
    with suppress(ProjectNameCollision):
        management.create_project(
            name=f"{neptune_workspace}/{wandb_project_name}",
            description=f"Exported from {wandb_project.url}",
        )

    with neptune.init_project(
        project=f"{neptune_workspace}/{wandb_project_name}"
    ) as neptune_project:
        # Log W&B project URL to Neptune
        neptune_project["wandb_url"] = wandb_project.url

        wandb_runs = [run for run in client.runs(f"{wandb_entity}/{wandb_project.name}")]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_run = {
                executor.submit(copy_run, wandb_run, wandb_project_name): wandb_run
                for wandb_run in wandb_runs
            }

            for future in tqdm(
                as_completed(future_to_run),
                total=len(future_to_run),
                desc=f"Copying {wandb_project_name} runs",
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(
                        f"Failed to copy {future_to_run[future]} due to exception:\n{e}"
                    )


# %%
# sourcery skip: identity-comprehension

try:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_project = {
            executor.submit(copy_project, wandb_project): wandb_project
            for wandb_project in wandb_projects
            if wandb_project.name in selected_projects
        }

        for future in tqdm(
            as_completed(future_to_project),
            total=len(future_to_project),
            desc="Copying projects",
        ):
            try:
                future.result()
            except Exception as e:
                logger.exception(
                    f"Failed to copy {future_to_project[future]} due to exception:\n{e}"
                )

    logger.info("Copy complete!")
    print("\nDone!")

except Exception as e:
    logger.exception(f"Copy failed due to exception:\n{e}")
    print("\nError!")
    raise e

finally:
    logging.shutdown()
    print(f"Check logs at {log_filename}\n")
