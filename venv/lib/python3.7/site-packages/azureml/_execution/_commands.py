# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# pylint: disable=line-too-long
from __future__ import print_function

import collections
import json
import os
import shutil
import subprocess
import time
import urllib3
import zipfile
import uuid
from multiprocessing.dummy import Pool
import sys
import platform
import requests
import re
import logging

import azureml._vendor.ruamel.yaml as ruamelyaml
from azureml._base_sdk_common import _ClientSessionId
from azureml._base_sdk_common.user_agent import get_user_agent
from azureml._base_sdk_common.common import get_project_files, get_run_config_dir_name
from azureml._base_sdk_common.common import normalize_windows_paths, give_warning
from azureml._base_sdk_common.common import RUNCONFIGURATION_EXTENSION, COMPUTECONTEXT_EXTENSION
from azureml._base_sdk_common.process_utilities import start_background_process
from azureml._base_sdk_common.utils import create_session_with_retry
from azureml._file_utils.file_utils import get_directory_size
from azureml.core._serialization_utils import _serialize_to_dict
from azureml._project.ignore_file import get_project_ignore_file
from azureml._restclient.run_client import RunClient
from azureml._restclient.clientbase import ClientBase
from azureml._restclient.snapshots_client import SnapshotsClient
from azureml._restclient.models.create_run_dto import CreateRunDto
from azureml.exceptions import ExperimentExecutionException, UserErrorException

from backports import tempfile as temp_dir_back

_one_mb = 1024 * 1024
_num_max_mbs = 25
_max_zip_size_bytes = _num_max_mbs * _one_mb

module_logger = logging.getLogger(__name__)


def prepare_compute_target(project_object, run_config_object, check=False, run_id=None, parent_run_id=None):
    """
    API to prepare a target for an experiment run.
    :param project_object: The project object.
    :type project_object: azureml.core.project.Project
    :param run_config_object: The run configuration object.
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :param check: True if it is only a prepare check operation, which will not do an actual prepare.
    :type check: bool
    :param run_id: A user specified run id.
    :type run_id: str
    :param parent_run_id: A user specified parent run id.
    :type parent_run_id: str
    :return: A run object except in only prepare check operation.
    :rtype: azureml.core.script_run.ScriptRun or bool in case of check=True
    """
    return start_run(project_object, run_config_object,
                     check=check, run_id=run_id, parent_run_id=parent_run_id, prepare_only=True)


# TODO - Delete injected files and all its references once automl moves to using the actual API.
def start_run(project_object, run_config_object,
              run_id=None, injected_files=None, telemetry_values=None,
              parent_run_id=None, prepare_only=False, check=False):
    """
    Start an experiment run for a project.
    Returns a run object.
    :param project_object: Project object.
    :type project_object: azureml.core.project.Project
    :param run_config_object: The run configuration object.
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :param run_id: A users specified run id.
    :type run_id: str
    :param injected_files:
    :type injected_files: dict
    :param telemetry_values: Telemetry property bag.
    :type telemetry_values: dict
    :param prepare_only: Whether this is a preparation run.
    :type prepare_only: bool
    :param check: True if it is only a prepare check operation, which will not do an actual prepare.
    :type check: bool
    :return: A run object.
    :rtype: azureml.core.script_run.ScriptRun
    """

    setup = _setup_run(project_object, run_config_object,
                       injected_files=injected_files, prepare_only=prepare_only)
    custom_target_dict = setup.custom_target_dict

    # Collect common telemetry information for client side.
    telemetry_values = _get_telemetry_values(telemetry_values) if not prepare_only else None

    """Submits an experiment."""
    if not run_id:
        run_id = _get_project_run_id(project_object)

    shared_start_run_kwargs = {"custom_target_dict": custom_target_dict,
                               "run_id": run_id,
                               "prepare_only": prepare_only,
                               "injected_files": injected_files,
                               "parent_run_id": parent_run_id}
    if telemetry_values is not None:
        shared_start_run_kwargs["telemetry_values"] = telemetry_values

    if _is_local_target(run_config_object.target, custom_target_dict):
        if prepare_only and check:
            raise ExperimentExecutionException("Can not check preparation of local targets")
        return _start_internal_local_cloud(project_object, run_config_object,
                                           **shared_start_run_kwargs)
    else:
        return _start_internal(project_object, run_config_object, prepare_check=check,
                               **shared_start_run_kwargs)


def _setup_run(project_object, run_config_object, prepare_only=None, injected_files=None):
    """
    Setup run
    :param project_object: Project object.
    :type project_object: azureml.core.project.Project
    :param run_config_object: The run configuration object.
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :param prepare_only: prepare_only=True for only prepare operation.
    :type prepare_only: bool
    :param injected_files:
    :type injected_files: dict
    :return:
    """
    configuration_path = _get_configuration_path(project_object, run_config_object._name,
                                                 config_type=RUNCONFIGURATION_EXTENSION)
    configuration_path = normalize_windows_paths(configuration_path)

    target = run_config_object.target

    if not target:
        raise ExperimentExecutionException("Must specify a target either through run configuration or arguments.")

    if target == "amlcompute" and run_config_object.amlcompute.vm_size is None:
        raise ExperimentExecutionException("Must specify VM size for the compute target to be created.")

    custom_target_dict = _get_custom_target_dict(project_object, target)

    if run_config_object.script:
        full_script_path = os.path.normpath(os.path.join(project_object.project_directory, run_config_object.script))
        if os.path.isfile(full_script_path):
            run_config_object.script = normalize_windows_paths(
                os.path.relpath(full_script_path, project_object.project_directory))
        else:
            raise ExperimentExecutionException("{} script path doesn't exist. "
                                               "The script should be inside the project "
                                               "folder".format(full_script_path))

    setup = collections.namedtuple("setup", ["custom_target_dict", "configuration_path"])
    return setup(custom_target_dict=custom_target_dict, configuration_path=configuration_path)


def _serialize_run_config_to_dict(run_config_object):
    result = _serialize_to_dict(run_config_object)
    return result


def _start_internal_local_cloud(project_object, run_config_object,
                                prepare_only=False, custom_target_dict=None, run_id=None,
                                injected_files=None, telemetry_values=None, parent_run_id=None):
    """
    :param project_object: Project object
    :type project_object: azureml.core.project.Project
    :param run_config_object: The run configuration object.
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :param prepare_only:
    :param custom_target_dict:
    :param run_id:
    :param injected_files:
    :type injected_files: dict
    :param telemetry_values:
    :param parent_run_id:
    :return: azureml.core.script_run.ScriptRun
    """
    service_context = project_object.workspace.service_context
    snapshots_client = SnapshotsClient(service_context)

    service_address = service_context._get_execution_url()
    #  TODO move this into project or experiment to avoid code dup
    service_arm_scope = "{}/experiments/{}".format(service_context._get_workspace_scope(), project_object.history.name)
    auth_header = project_object.workspace._auth_object.get_authentication_header()
    thread_pool = Pool(1)

    snapshot_async = None
    if run_config_object.history.snapshot_project:
        snapshot_async = thread_pool.apply_async(
            snapshots_client.create_snapshot,
            (project_object.project_directory,))

    _run_config_modification(run_config_object)

    # Check size of config folder
    if get_directory_size(
            project_object.project_directory, _max_zip_size_bytes, include_function=_include) > _max_zip_size_bytes:
        error_message = "====================================================================\n" \
                        "\n" \
                        "Your configuration directory exceeds the limit of {0} MB.\n" \
                        "Please see http://aka.ms/aml-largefiles on how to work with large files.\n" \
                        "\n" \
                        "====================================================================\n" \
                        "\n".format(_max_zip_size_bytes / _one_mb)
        raise ExperimentExecutionException(error_message)

    with temp_dir_back.TemporaryDirectory() as temporary:
        archive_path = os.path.join(temporary, "aml_config.zip")
        archive_path_local = os.path.join(temporary, "temp_project.zip")

        project_temp_dir = _get_project_temporary_directory(run_id)
        os.mkdir(project_temp_dir)

        # We send only aml_config zip to service and copy only necessary files to temp dir
        ignore_file = get_project_ignore_file(project_object.project_directory)
        _make_zipfile_include(project_object, archive_path, _include)
        _make_zipfile_exclude(project_object, archive_path_local, ignore_file.is_file_excluded)

        # Inject files into the user's project'
        if injected_files:
            _add_files_to_zip(archive_path, injected_files)

        # Copy current project dir to temp/azureml-runs folder.
        zip_ref = zipfile.ZipFile(archive_path_local, "r")
        zip_ref.extractall(project_temp_dir)
        zip_ref.close()

        # TODO Missing driver arguments, job_name

        with open(archive_path, "rb") as archive:
            definition = {
                "TargetDetails": custom_target_dict,
                "Configuration": _serialize_run_config_to_dict(run_config_object),
                "TelemetryValues": telemetry_values}
            if parent_run_id is not None:
                definition["ParentRunId"] = parent_run_id

            files = [
                ("files", ("definition.json", json.dumps(definition, indent=4, sort_keys=True))),
                ("files", ("aml_config.zip", archive))]

            headers = _get_common_headers()

            # Merging the auth header.
            headers.update(auth_header)

            uri = service_address + "/execution/v1.0" + service_arm_scope
            if prepare_only:
                uri += "/localprepare"
            else:
                uri += "/localrun"

            # Unfortunately, requests library does not take Queryparams nicely.
            # Appending run_id_query to the url for service to extract from it.
            run_id_query = urllib3.request.urlencode({"runId": run_id})
            uri += "?" + run_id_query

            response = ClientBase._execute_func(requests.post, uri, files=files, headers=headers)
            _raise_request_error(response, "starting run")

            invocation_zip_path = os.path.join(project_temp_dir, "invocation.zip")
            with open(invocation_zip_path, "wb") as file:
                file.write(response.content)

            with zipfile.ZipFile(invocation_zip_path, "r") as zip_ref:
                zip_ref.extractall(project_temp_dir)

            try:
                _invoke_command(project_temp_dir)
            except subprocess.CalledProcessError as ex:
                raise ExperimentExecutionException(ex.output)

            snapshot_id = snapshot_async.get() if snapshot_async else None
            thread_pool.close()

            return _get_run_details(project_object, run_config_object, run_id,
                                    snapshot_id=snapshot_id)


# TODO: Need to update the documentation.
# This function is used across prepare and start functions.
# For prepare, there are two parameters: 1) prepare_only and 2) prepare_check. If prepare_only=True, then only
# prepare operation is run, and actual experiment run is not run. If prepare_only=False, then both the prepare
# operation and experiment is run. If prepare_only=True and prepare_check=False, then the an experiment prepare
# is done. If prepare_only=True and prepare_check=True, then only prepare status is checked and an actual prepare is
# not performed.
# prepare_only=False and prepare_check=True has no effect, the experiment wil be run and prepare will be performed
# if needed.
# If prepare_only=True and prepare_check=True, then tracked_run, async, wait arguments have no effect.
def _start_internal(project_object, run_config_object,
                    prepare_only=False, prepare_check=False,
                    custom_target_dict=None, run_id=None,
                    injected_files=None, telemetry_values=None,
                    parent_run_id=None):
    """
    :param project_object: Project object
    :type project_object: azureml.core.project.Project
    :param run_config_object: The run configuration object.
    :param run_config_object: azureml.core.runconfig.RunConfiguration
    :param prepare_only:
    :param prepare_check:
    :param custom_target_dict:
    :param run_id:
    :param injected_files:
    :type injected_files: dict
    :param telemetry_values:
    :param parent_run_id:
    :return: azureml.core.script_run.ScriptRun or bool if prepare_check=True
    """
    service_context = project_object.workspace.service_context
    snapshots_client = SnapshotsClient(service_context)

    service_address = service_context._get_execution_url()
    service_arm_scope = "{}/experiments/{}".format(service_context._get_workspace_scope(), project_object.history.name)

    if run_config_object.credential_passthrough:
        aml_client_token = project_object.workspace._auth_object._get_azureml_client_token()
        auth_header = {"Authorization": "Bearer " + aml_client_token}
    else:
        auth_header = project_object.workspace._auth_object.get_authentication_header()
    thread_pool = Pool(1)
    ignore_file = get_project_ignore_file(project_object.project_directory)

    snapshot_async = None
    execute_with_zip = False

    directory_size = get_directory_size(
        project_object.project_directory,
        _max_zip_size_bytes,
        exclude_function=ignore_file.is_file_excluded)

    if directory_size >= _max_zip_size_bytes:
        give_warning("Submitting {} directory for run. "
                     "The size of the directory >= {} MB, "
                     "so it can take a few minutes.".format(project_object.project_directory, _num_max_mbs))

    if run_config_object.history.snapshot_project:
        snapshot_async = thread_pool.apply_async(
            snapshots_client.create_snapshot,
            (project_object.project_directory,))

        # These can be set by users in case we have any issues with zip/snapshot and need to force a specific path
        force_execute_snapshot = os.environ.get("AML_FORCE_EXECUTE_WITH_SNAPSHOT")
        force_execute_zip = os.environ.get("AML_FORCE_EXECUTE_WITH_ZIP")

        if force_execute_zip and not force_execute_snapshot:
            execute_with_zip = True
            module_logger.debug("Executing with zip file.")
        else:
            module_logger.debug("Executing with snapshot.")

    _run_config_modification(run_config_object)

    temporary = None
    archive = None
    try:
        if execute_with_zip:
            temporary = temp_dir_back.TemporaryDirectory()
            archive_path = os.path.join(temporary.name, "project.zip")
            _make_zipfile_exclude(project_object, archive_path, ignore_file.is_file_excluded)

            # Inject files into the user's project'
            if injected_files:
                _add_files_to_zip(archive_path, injected_files)
            archive = open(archive_path, "rb")

        headers = _get_common_headers()
        # Merging the auth header.
        headers.update(auth_header)

        uri = service_address
        api_prefix = ""

        if execute_with_zip:
            if not prepare_only:
                api_prefix = "start"
        else:
            api_prefix = "snapshot"

        if prepare_only:
            if prepare_check:
                uri += "/execution/v1.0" + service_arm_scope + "/{}checkprepare".format(api_prefix)
            else:
                uri += "/execution/v1.0" + service_arm_scope + "/{}prepare".format(api_prefix)

        else:
            uri += "/execution/v1.0" + service_arm_scope + "/{}run".format(api_prefix)

        run_id_query = urllib3.request.urlencode({"runId": run_id})

        uri += "?" + run_id_query

        snapshot_id = snapshot_async.get() if snapshot_async else None
        thread_pool.close()

        definition = {
            "TargetDetails": custom_target_dict,
            "Configuration": _serialize_run_config_to_dict(run_config_object),
            "TelemetryValues": telemetry_values
        }
        if parent_run_id is not None:
            definition["ParentRunId"] = parent_run_id

        if execute_with_zip:
            if prepare_only:
                files = [
                    ("files", ("definition.json", json.dumps(definition, indent=4, sort_keys=True))),
                    ("files", ("project.zip", archive))]
            else:
                files = [
                    ("runDefinitionFile", ("definition.json", json.dumps(definition, indent=4, sort_keys=True))),
                    ("projectZipFile", ("project.zip", archive))]

            response = ClientBase._execute_func(requests.post, uri, files=files, headers=headers)
        else:
            definition["SnapshotId"] = snapshot_id

            response = ClientBase._execute_func(requests.post, uri, json=definition, headers=headers)

        _raise_request_error(response, "starting run")

    finally:
        if archive:
            archive.close()
        if temporary:
            temporary.cleanup()

    result = response.json()
    if prepare_only and prepare_check:
        return result["environmentPrepared"]

    return _get_run_details(project_object, run_config_object, result["runId"],
                            snapshot_id=snapshot_id)


def _get_run_details(project_object, run_config_object, run_id, snapshot_id=None):
    """
    Returns a run object or bool in case prepare_check=True
    :param project_object:
    :type project_object: azureml._project.project.Project
    :param run_config_object:
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :return:
    :rtype: azureml.core.script_run.ScriptRun
    """
    from azureml.core.script_run import ScriptRun
    client = RunClient(project_object.workspace.service_context, project_object.history.name, run_id,
                       experiment_id=project_object.history.id)

    run_properties = {
        "ContentSnapshotId": snapshot_id,
    }
    create_run_dto = CreateRunDto(run_id, properties=run_properties)
    run_dto = client.patch_run(create_run_dto)
    experiment = project_object.history
    return ScriptRun(experiment, run_id,
                     directory=project_object.project_directory,
                     _run_config=run_config_object, _run_dto=run_dto)


def run_status(project_object, run_config_object, run_id):
    """
    Retrieves the run status.
    :param project_object: Project object.
    :type project_object: azureml.core.project.Project
    :param run_id: The run id
    :type run_id: str
    :param run_config_object: The run configuration object.
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :return: The status in dict format.
    :rtype dict:
    """

    return _get_status_from_history(project_object, run_id)


def clean(project_object, run_config_object, run_id=None, all=False):
    """
    Removes files corresponding to azureml run(s) from a target.
    Doesn't delete docker images for the local docker target.
    :param project_object: Project object
    :type project_object: azureml.core.project.Project
    :param run_config_object: The run configuration object
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :param run_id: The run id for the experiment whose files should be removed.
    :type run_id: str
    :param all: all=True, removes files for all azureml runs.
    :return: List of files deleted.
    :rtype: list
    """
    def remove_docker_image():
        return "Docker images used by AzureML have not been removed; they can be deleted with \"docker rmi\""

    custom_target_dict = _get_custom_target_dict(project_object, run_config_object.target)

    if all and run_id:
        raise UserErrorException("Error: Specify either --all or specific run to clean "
                                 "with --run <run_id>, not both.")

    if _is_local_target(run_config_object.target, custom_target_dict):
        if run_id is not None:
            output_list = []
            project_temp_dir = _get_project_temporary_directory(run_id)
            if os.path.exists(project_temp_dir):
                shutil.rmtree(project_temp_dir)
                # TODO: We are adding some text, instead of returning the directory directly?
                output_list.append("Removed temporary run directory {0}".format(os.path.abspath(project_temp_dir)))
                return output_list
            else:
                raise ExperimentExecutionException("Temporary directory for this run does not exist.")
        else:
            output_list = []
            import tempfile
            azureml_temp_dir = os.path.join(tempfile.gettempdir(), "azureml_runs")
            temp_runs = os.listdir(azureml_temp_dir)
            for run_id in temp_runs:
                project_temp_dir = _get_project_temporary_directory(run_id)
                shutil.rmtree(project_temp_dir)
                output_list.append("Removed temporary run directory {0}".format(os.path.abspath(project_temp_dir)))
            if custom_target_dict["type"] in ["docker", "localdocker"]:
                output_list.append(remove_docker_image())
            if custom_target_dict["type"] == "local":
                dot_azureml_dir = os.path.join(os.path.expanduser("~"), ".azureml")
                envs_dir = os.path.join(dot_azureml_dir, "envs")
                locks_dir = os.path.join(dot_azureml_dir, "locks")
                shutil.rmtree(envs_dir)
                shutil.rmtree(locks_dir)
                output_list.append("Removed managed environment directory {0}".format(os.path.abspath(envs_dir)))
            return output_list
    else:

        body = {
            "RunId": run_id if run_id else "null",
            "Target": run_config_object.target,
            "CustomTarget": custom_target_dict}

        service_context = project_object.workspace.service_context
        uri = "{}/execution/v1.0{}/experiments/{}/clean?all={}".format(service_context._get_execution_url(),
                                                                       service_context._get_workspace_scope(),
                                                                       project_object.history.name,
                                                                       str(all))

        auth_header = project_object.workspace._auth_object.get_authentication_header()
        headers = _get_common_headers()

        headers.update(auth_header)

        session = create_session_with_retry()
        response = session.post(uri, json=body, headers=headers)
        _raise_request_error(response, "Deleting vienna run(s).")

        clean_list = response.json()
        return clean_list


def _raise_request_error(response, action="calling backend service"):
    if response.status_code >= 400:
        from azureml._base_sdk_common.common import get_http_exception_response_string
        # response.text is a JSON from execution service.
        response_message = get_http_exception_response_string(response)
        raise ExperimentExecutionException(response_message)


def _include(path):
    run_config_dir_name = get_run_config_dir_name(path)
    return any(folder in path for folder in [run_config_dir_name])


# Adapted from shutil._make_zipfile with an added exclusion function.
def _make_zipfile_exclude(project_object, zip_filename, exclude_function):
    base_dir = project_object.project_directory

    with zipfile.ZipFile(zip_filename, "w") as zf:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            relative_dirpath = os.path.relpath(dirpath, base_dir)
            for name in sorted(dirnames):
                full_path = os.path.normpath(os.path.join(dirpath, name))
                relative_path = os.path.normpath(os.path.join(relative_dirpath, name))
                if not exclude_function(full_path):
                    zf.write(full_path, relative_path)
            for name in filenames:
                full_path = os.path.normpath(os.path.join(dirpath, name))
                relative_path = os.path.normpath(os.path.join(relative_dirpath, name))
                if not exclude_function(full_path):
                    if os.path.isfile(full_path):
                        zf.write(full_path, relative_path)


def _make_zipfile_include(project_object, zip_filename, include_function):
    base_dir = project_object.project_directory

    with zipfile.ZipFile(zip_filename, "w") as zf:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            relative_dirpath = os.path.relpath(dirpath, base_dir)
            for name in sorted(dirnames):
                full_path = os.path.normpath(os.path.join(dirpath, name))
                relative_path = os.path.normpath(os.path.join(relative_dirpath, name))
                if include_function(full_path):
                    zf.write(full_path, relative_path)
            for name in filenames:
                full_path = os.path.normpath(os.path.join(dirpath, name))
                relative_path = os.path.normpath(os.path.join(relative_dirpath, name))
                if include_function(full_path):
                    if os.path.isfile(full_path):
                        zf.write(full_path, relative_path)


def _is_local_target(target_name, custom_target_dict):
    # Local is a magic compute target name; no config file or MLC object is required.
    if target_name == "local":
        return True

    if not custom_target_dict:
        return False

    return custom_target_dict["type"] in ["local", "docker", "localdocker"]


def _get_status_from_history(project_object, run_id):
    """
    Queries the execution service for status of an experiment from history service.
    Returns the JSONObject returned by the service.
    :param project_object:
    :type project_object: azureml.core.project.Project
    :param run_id:
    :return:
    :rtype: dict
    """
    run_client = RunClient(project_object.workspace.service_context, project_object.history.name, run_id,
                           experiment_id=project_object.history.id)
    return run_client.get_runstatus(headers=_get_common_headers()).as_dict()


def _get_content_from_uri(uri, session=None, timeout=None):
    """
    Queries the artifact service for an artifact and returns it's contents
    Returns the Response text by the service
    :return:
    :rtype: dict
    """

    if session is None:
        session = create_session_with_retry()
    response = session.get(uri, timeout=timeout)

    # Match old behavior from execution service's status API.
    if response.status_code == 404:
        return ""

    _raise_request_error(response, "Retrieving content from " + uri)
    return response.text


def _get_common_headers():
    headers = {
        "User-Agent": get_user_agent(),
        "x-ms-client-session-id": _ClientSessionId,
        "x-ms-client-request-id": str(uuid.uuid4())
    }
    return headers


def _get_configuration_path(project_object, name, config_type):
    files = get_project_files(project_object.project_directory, config_type)
    if name not in files:
        return None

    return files[name]


def _read_run_configuration(path):
    # TODO: Notebook calls this.
    with open(path, "r") as file:
        return ruamelyaml.load(file.read())


def _get_custom_target_dict(project_object, target):
    project_targets = get_project_files(project_object.project_directory, COMPUTECONTEXT_EXTENSION)
    if target in project_targets:
        path_to_target = project_targets[target]
        with open(os.path.join(project_object.project_directory,
                               path_to_target), "r") as file:
            return ruamelyaml.load(file.read())

    return None


def _add_files_to_zip(archive_path, injected_files):
    try:
        archive = zipfile.ZipFile(archive_path, "a", zipfile.ZIP_DEFLATED)
        for file_path in injected_files.keys():
            dest_path = injected_files[file_path]
            archive.write(file_path, dest_path)
    finally:
        archive.close()


def _invoke_command(project_temp_dir):
    # Delete the zip file from tempdir - that comes from the service.
    invocation_zip_file = os.path.join(project_temp_dir, "Invocation.zip")
    if os.path.isfile(invocation_zip_file):
        os.remove(invocation_zip_file)

    if os.name == "nt":
        invocation_script = os.path.join(project_temp_dir, "azureml-setup", "Invocation.bat")
        invoke_command = ["cmd.exe", "/c", "{0}".format(invocation_script)]
    else:
        invocation_script = os.path.join(project_temp_dir, "azureml-setup", "Invocation.sh")
        subprocess.check_output(["chmod", "+x", invocation_script])
        invoke_command = ["/bin/bash", "-c", "{0}".format(invocation_script)]

    env = os.environ.copy()
    env.pop("AZUREML_TARGET_TYPE", None)

    start_background_process(invoke_command, cwd=project_temp_dir, env=env)


def _get_project_temporary_directory(run_id):
    import tempfile
    azureml_temp_dir = os.path.join(tempfile.gettempdir(), "azureml_runs")
    if not os.path.isdir(azureml_temp_dir):
        os.mkdir(azureml_temp_dir)

    project_temp_dir = os.path.join(azureml_temp_dir, run_id)
    return project_temp_dir


def _get_project_run_id(project_object):
    project_name = project_object.experiment.name
    return _generate_run_id_from_experiment_name(project_name)


def _generate_run_id_from_experiment_name(experiment_name):
    run_id = experiment_name + "_" + str(int(time.time())) + "_" + str(uuid.uuid4())[:8]
    return run_id


def _get_telemetry_values(telemetry_values):
    if telemetry_values is None:
        telemetry_values = {}

    # Collect common client info
    telemetry_values["amlClientOSVersion"] = platform.platform()
    telemetry_values["amlClientPythonVersion"] = sys.version

    # notebookvm related...
    notebook_instance_file = "/mnt/azmnt/.nbvm"
    notebook_instance_config = {}
    is_notebook_vm = False
    if os.path.exists(notebook_instance_file) and os.path.isfile(notebook_instance_file):
        is_notebook_vm = True
        try:
            # parse key=value lines into a dictionary via regex
            envre = re.compile(r'''^([^\s=]+)=(?:[\s"']*)(.+?)(?:[\s"']*)$''')
            with open(notebook_instance_file) as nbvm_variables:
                for line in nbvm_variables:
                    match = envre.match(line)
                    if match is not None:
                        notebook_instance_config[match.group(1)] = match.group(2)
        except Exception:
            pass

    telemetry_values["notebookVM"] = is_notebook_vm
    if "instance" in notebook_instance_config:
        telemetry_values["notebookInstanceName"] = notebook_instance_config["instance"]
    if "domainsuffix" in notebook_instance_config:
        telemetry_values["notebookInstanceDomainSuffix"] = notebook_instance_config["domainsuffix"]

    # Do not override client type if it already exists
    telemetry_values.setdefault("amlClientType", "azureml-sdk-core")

    return telemetry_values


def _run_config_modification(run_config):
    if isinstance(run_config.command, list):
        run_config.command = " ".join(run_config.command)


class GitRepositoryInfo(object):
    def __init__(self, git_repository_url, branch_name=None, commit_id=None, working_directory=None):
        self.git_repository_url = git_repository_url
        self.branch_name = branch_name
        self.commit_id = commit_id
        self.working_directory = working_directory


def _start_git_snapshot_run(experiment, git_repository_info, run_config_object, run_id=None,
                            parent_run_id=None):
    """
    This is used only for tests now. git snapshot run is not publically exposed through v1 sdk.
    Contracts for Git snapshot and Git run submit is not yet decided for v1 SDK.
    :param experiment:
    :type experiment: azureml.core.experiment.Experiment
    :param git_repository_info:
    :type git_repository_info: GitRepositoryInfo
    :param run_config_object:
    :type run_config_object: azureml.core.runconfig.RunConfiguration
    :param run_id:
    :type run_id: str
    :param parent_run_id:
    :type parent_run_id: str
    :return: A run object.
    :rtype: azureml.core.script_run.ScriptRun
    """
    service_context = experiment.workspace.service_context
    snapshots_client = SnapshotsClient(service_context)
    snapshot_id = snapshots_client.create_git_snapshot(
        git_repository_info.git_repository_url,
        branch_name=git_repository_info.branch_name,
        commit_id=git_repository_info.commit_id,
        working_directory=git_repository_info.working_directory)

    if not run_id:
        run_id = _generate_run_id_from_experiment_name(experiment.name)

    base_url = service_context._get_execution_url()
    experiment_scope = "{}/experiments/{}".format(service_context._get_workspace_scope(), experiment.name)

    auth_header = experiment.workspace._auth_object.get_authentication_header()

    _run_config_modification(run_config_object)

    headers = _get_common_headers()
    # Merging the auth header.
    headers.update(auth_header)

    api_url = "{}/execution/v1.0{}/snapshotrun".format(base_url, experiment_scope)

    run_id_query = urllib3.request.urlencode({"runId": run_id})

    api_url += "?" + run_id_query

    serialized_run_config = _serialize_run_config_to_dict(run_config_object)
    definition = {
        "configuration": serialized_run_config,
        "snapshotId": snapshot_id
    }
    if parent_run_id is not None:
        definition["ParentRunId"] = parent_run_id

    response = ClientBase._execute_func(requests.post, api_url, json=definition, headers=headers)

    _raise_request_error(response, "starting git snapshot run")

    result = response.json()

    # Creating a Project for _get_run_details() to work.
    from azureml._project.project import Project
    project_object = Project(experiment=experiment, auth=experiment.workspace._auth_object,
                             _disable_service_check=True)

    return _get_run_details(project_object, run_config_object, result["runId"],
                            snapshot_id=snapshot_id)
