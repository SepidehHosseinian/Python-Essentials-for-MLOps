# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Access ArtifactsClient"""

import os
import requests
import json
import uuid
import tempfile
import time

from azureml._async import TaskQueue
from azureml._file_utils import download_file, get_path_size, upload_blob_from_stream

from .workspace_client import WorkspaceClient
from .constants import SNAPSHOT_MAX_FILES, ONE_MB, SNAPSHOT_MAX_SIZE_BYTES

from azureml._base_sdk_common.common import get_http_exception_response_string
from azureml._base_sdk_common.merkle_tree import DirTreeJsonEncoder, DirTreeJsonEncoderV2, create_merkletree
from azureml._base_sdk_common.merkle_tree_differ import compute_diff
from azureml._base_sdk_common.project_snapshot_cache import ContentSnapshotCache
from azureml._base_sdk_common.snapshot_dto import SnapshotDto
from azureml._base_sdk_common.tracking import global_tracking_info_registry
from azureml._base_sdk_common.utils import create_session_with_retry
from azureml._project.ignore_file import get_project_ignore_file
from azureml.exceptions import SnapshotException, ProjectSystemException

MAX_FILES_SNAPSHOT_UPLOAD_PARALLEL = 50
RETRY_LIMIT = 3
BACKOFF_START = 2
# Timeout Documentation:
# https://docs.microsoft.com/en-us/rest/api/storageservices/Setting-Timeouts-for-Blob-Service-Operations
TIMEOUT = 30
AZUREML_SNAPSHOT_DEFAULT_TIMEOUT = 300
FILE_NODES_KEY = "fileNodes"


class SnapshotsClient(WorkspaceClient):
    """
    Snapshot client class

    :param service_context:
    :type service_context: azureml._restclient.service_context.ServiceContext
    """
    def __init__(self, *args, **kwargs):
        super(SnapshotsClient, self).__init__(*args, **kwargs)
        self._cache = ContentSnapshotCache(self._service_context)

    def _validate_snapshot_size(self, file_or_folder_path, exclude_function, raise_on_validation_failure):
        size = get_path_size(file_or_folder_path,
                             size_limit=SNAPSHOT_MAX_SIZE_BYTES, exclude_function=exclude_function)

        if size > SNAPSHOT_MAX_SIZE_BYTES:
            error_message = "====================================================================\n" \
                            "\n" \
                            "While attempting to take snapshot of {}\n" \
                            "Your total snapshot size exceeds the limit of {} MB.\n" \
                            "Please see http://aka.ms/aml-largefiles on how to work with large files.\n" \
                            "\n" \
                            "====================================================================\n" \
                            "\n".format(file_or_folder_path, SNAPSHOT_MAX_SIZE_BYTES / ONE_MB)
            if raise_on_validation_failure:
                raise SnapshotException(error_message)
            else:
                self._logger.warning(error_message)

    def _validate_snapshot_file_count(self, file_or_folder_path, file_number, raise_on_validation_failure):
        if file_number > SNAPSHOT_MAX_FILES and not os.environ.get("AML_SNAPSHOT_NO_FILE_LIMIT"):
            error_message = "====================================================================\n" \
                            "\n" \
                            "While attempting to take snapshot of {}\n" \
                            "Your project exceeds the file limit of {}.\n" \
                            "\n" \
                            "====================================================================\n" \
                            "\n".format(file_or_folder_path, SNAPSHOT_MAX_FILES)
            if raise_on_validation_failure:
                raise SnapshotException(error_message)
            else:
                print(error_message)

    def create_snapshot(self, file_or_folder_path, retry_on_failure=True, raise_on_validation_failure=True):
        ignore_file = get_project_ignore_file(file_or_folder_path)
        exclude_function = ignore_file.is_file_excluded

        self._validate_snapshot_size(file_or_folder_path, exclude_function, raise_on_validation_failure)

        # Get the previous snapshot for this project
        try:
            parent_root, parent_snapshot_id = self._cache.get_latest_snapshot()
        except json.decoder.JSONDecodeError:
            # Removing the cache file if found corrupted
            self._cache.remove_latest()
            parent_root, parent_snapshot_id = self._cache.get_latest_snapshot()

        # Compute the dir tree for the current working set
        curr_root = create_merkletree(file_or_folder_path, exclude_function)

        # Compute the diff between the two dirTrees
        entries = compute_diff(parent_root, curr_root)
        dir_tree_file_contents = json.dumps(curr_root, cls=DirTreeJsonEncoder)

        # If there are no changes, just return the previous snapshot_id
        if not len(entries):
            return parent_snapshot_id

        entries_to_send = [entry for entry in entries if
                           (entry.operation_type == 'added' or entry.operation_type == 'modified') and entry.is_file]
        self._validate_snapshot_file_count(file_or_folder_path, len(entries_to_send), raise_on_validation_failure)

        # Git metadata
        snapshot_properties = global_tracking_info_registry.gather_all(file_or_folder_path)

        new_snapshot_id = str(uuid.uuid4())

        headers = {'Content-Type': 'application/json; charset=UTF-8'}
        headers.update(self.auth.get_authentication_header())

        with create_session_with_retry() as session:
            revision_list = self._upload_snapshot_files(entries_to_send, file_or_folder_path, exclude_function)

            create_data = {"ParentSnapshotId": parent_snapshot_id, "Tags": None, "Properties": snapshot_properties}
            create_data.update({"DirTreeNode": curr_root})
            create_data.update({"FileRevisionList": {"FileNodes": revision_list}})

            data = json.dumps(create_data, cls=DirTreeJsonEncoderV2)
            encoded_data = data.encode('utf-8')

            url = self._service_context._get_project_content_url() + "/content/v2.0" + \
                self._service_context._get_workspace_scope() + "/snapshots/" + new_snapshot_id
            response = self._execute_with_base_arguments(
                session.post, url, data=encoded_data, headers=headers)

            if response.status_code >= 400:
                if retry_on_failure:
                    self._cache.remove_latest()
                    return self.create_snapshot(file_or_folder_path, retry_on_failure=False)
                else:
                    raise SnapshotException(get_http_exception_response_string(response))

        # Update the cache
        snapshot_dto = SnapshotDto(dir_tree_file_contents, new_snapshot_id)
        self._cache.update_cache(snapshot_dto)
        return new_snapshot_id

    def get_rest_client(self, user_agent=None):
        return self._service_context._get_project_content_restclient(user_agent=user_agent)

    def restore_snapshot(self, snapshot_id, path):
        headers = self.auth.get_authentication_header()

        with create_session_with_retry() as session:
            url = self._service_context._get_project_content_url() + "/content/v2.0" + \
                self.get_workspace_uri_path() + "/snapshots/" + snapshot_id + "/zip"
            response = self._execute_with_base_arguments(session.post, url, headers=headers)
            # This returns a sas url to blob store
            response.raise_for_status()

            if response.status_code not in (200, 202):
                message = "Error building snapshot zip. Code: {}\n: {}".format(response.status_code,
                                                                               response.text)
                raise Exception(message)

            response_content = response.content
            response_json = json.loads(response_content.decode('utf-8'))

            # This could be a 200 if this snapshot has already been zipped previously
            if response.status_code == 202:
                location = response_json["location"]
                response_content = self._get_snapshot_zip(snapshot_id, location)
                response_json = json.loads(response_content.decode('utf-8'))

            sas_url = response_json['zipSasUri']
            snapshot_file_name = str(snapshot_id) + '.zip'
            if path is None:
                path = tempfile.gettempdir()

            temp_path = os.path.join(path, snapshot_file_name)
            try:
                download_file(sas_url, temp_path, session=session)
            except requests.HTTPError as http_error:
                raise ProjectSystemException(http_error.strerror)

        return os.path.abspath(temp_path)

    def _upload_files_batch(self, file_nodes, file_or_folder_path, session):
        batch_size = MAX_FILES_SNAPSHOT_UPLOAD_PARALLEL
        results = []
        revision_list = []

        for i in range(0, len(file_nodes), batch_size):
            with TaskQueue(worker_pool=self._pool, flush_timeout_seconds=AZUREML_SNAPSHOT_DEFAULT_TIMEOUT,
                           _ident="snapshot_upload_files", _parent_logger=self._logger) as task_queue:
                batch_nodes = file_nodes[i:i + batch_size]

                def perform_upload(file_or_folder_path, file_name, upload_url, session):
                    with open(os.path.join(file_or_folder_path, file_name), "rb") as data:
                        return upload_blob_from_stream(data, upload_url, session=session, timeout=TIMEOUT,
                                                       backoff=BACKOFF_START, retries=RETRY_LIMIT)

                for node in batch_nodes:
                    file_name = node['fullName']
                    upload_url = node['blobUri']
                    task = task_queue.add(perform_upload, file_or_folder_path, file_name, upload_url, session)
                    results.append(task)

                    file_size = os.path.getsize(os.path.join(file_or_folder_path, file_name))
                    file_revision = {"FullName": file_name, "BlobUri": upload_url, "FileSize": file_size}
                    revision_list.append(file_revision)

        return revision_list, map(lambda task: task.wait(), results)

    def _upload_snapshot_files(self, entries_to_send, file_or_folder_path, exclude_function):
        headers = {'Content-Type': 'application/json; charset=UTF-8'}
        headers.update(self.auth.get_authentication_header())

        file_list = [entry.node_path for entry in entries_to_send]
        file_names = {"FileNames": file_list}

        data = json.dumps(file_names)
        encoded_data = data.encode('utf-8')

        with create_session_with_retry() as session:
            get_blobs_url = self._service_context._get_project_content_url() + "/content/v2.0" + \
                self._service_context._get_workspace_scope() + "/snapshots/getblob"
            get_blobs_response = self._execute_with_base_arguments(
                session.post, get_blobs_url, data=encoded_data, headers=headers)

            status = get_blobs_response.status_code
            if status not in (200, 202):
                message = "Error uploading snapshot files to storage. Code: {}\n: {}"\
                    .format(status, get_blobs_response.text)
                raise SnapshotException(message)

            response_data = get_blobs_response.content.decode('utf-8')
            blob_uri_dict = json.loads(response_data)
            if blob_uri_dict and FILE_NODES_KEY in blob_uri_dict:
                file_nodes = blob_uri_dict[FILE_NODES_KEY]
            elif blob_uri_dict:
                raise SnapshotException("No {} found in blob_uri_dict".format(FILE_NODES_KEY))
            else:
                raise SnapshotException("No fileNodes credentials found")

            revision_list = self._upload_files_batch(file_nodes, file_or_folder_path, session)
            return revision_list[0]

    def _get_snapshot_zip(self, snapshot_id, location):
        headers = self.auth.get_authentication_header()
        status = 202
        timeout_seconds = 60
        time_run = 0
        sleep_period = 5

        while (status == 202):
            if time_run + sleep_period > timeout_seconds:
                message = "Timeout on Get zip for snapshot : {}\n".format(snapshot_id)
                raise Exception(message)
            time_run += sleep_period
            time.sleep(sleep_period)
            with create_session_with_retry() as session:
                response = self._execute_with_base_arguments(session.get, location, headers=headers)
                # This returns a sas url to blob store
                response.raise_for_status()
                status = response.status_code
                if status not in (200, 202):
                    message = "Error Get snapshot zip. Code: {}\n: {}".format(status, response.text)
                    raise Exception(message)
        return response.content

    def create_git_snapshot(self, git_repository_url, branch_name=None, commit_id=None, working_directory=None):
        """
        Creates git snapshot and returns the snapshot id.
        :param git_repository_url:
        :type git_repository_url: str
        :param branch_name:
        :type branch_name: str
        :param commit_id:
        :type commit_id: str
        :param working_directory:
        :type working_directory: str
        :return: snapshot id
        :rtype: str
        """
        new_snapshot_id = str(uuid.uuid4())

        git_uri_with_extended_syntax = git_repository_url
        if working_directory:
            git_uri_with_extended_syntax += "?path=" + working_directory
        if commit_id:
            git_uri_with_extended_syntax += "#" + commit_id
        elif branch_name:
            git_uri_with_extended_syntax += "#" + branch_name

        create_snapshot_using_git_dto = {
            "uri": git_uri_with_extended_syntax
        }
        api_url = self._service_context._get_project_content_url() + "/content/v2.0" + \
            self._service_context._get_workspace_scope() + "/snapshots/uri/" + new_snapshot_id

        with create_session_with_retry() as session:
            response = session.post(api_url, json=create_snapshot_using_git_dto,
                                    headers=self.auth.get_authentication_header())

            if response.status_code >= 400:
                raise SnapshotException(get_http_exception_response_string(response))

        return new_snapshot_id
