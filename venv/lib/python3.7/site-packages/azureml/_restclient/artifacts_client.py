# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Access ArtifactsClient"""

import os

from io import IOBase

from msrest.exceptions import HttpOperationError

from azureml._async import TaskQueue
from azureml._file_utils import download_file, download_file_stream, upload_blob_from_stream

from azureml.exceptions import UserErrorException, AzureMLException, AzureMLAggregatedException
from .models.batch_artifact_container_sas_ingest_command import BatchArtifactContainerSasIngestCommand
from .models.artifact_path_dto import ArtifactPathDto
from .models.batch_artifact_create_command import BatchArtifactCreateCommand
from .workspace_client import WorkspaceClient

SUPPORTED_NUM_EMPTY_ARTIFACTS = 50
RETRY_LIMIT = 3
BACKOFF_START = 2
# Timeout Documentation:
# https://docs.microsoft.com/en-us/rest/api/storageservices/Setting-Timeouts-for-Blob-Service-Operations
TIMEOUT = 30

AZUREML_ARTIFACTS_TIMEOUT_ENV_VAR = "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"
AZUREML_ARTIFACTS_MIN_TIMEOUT = 300


class ArtifactsClient(WorkspaceClient):
    """
    Artifacts client class

    :param host: The base path for the server to call.
    :type host: str
    :param auth: Client authentication
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :param resource_group_name:
    :type resource_group_name: str
    :param workspace_name:
    :type workspace_name: str
    """

    def __init__(self, *args, **kwargs):
        super(ArtifactsClient, self).__init__(*args, **kwargs)
        batch_size = SUPPORTED_NUM_EMPTY_ARTIFACTS
        self.session = self._service_context._get_shared_session(pool_maxsize=batch_size)

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_artifacts_restclient(user_agent=user_agent)

    def create_empty_artifacts(self, origin, container, paths, datastore_name=None):
        """create empty artifacts"""
        if isinstance(paths, str):
            paths = [paths]
        artifacts = [ArtifactPathDto(path) for path in paths]
        batch_create_command = BatchArtifactCreateCommand(artifacts)
        res = self._execute_with_workspace_arguments(
            self._client.artifact.batch_create_empty_artifacts,
            origin,
            container,
            datastore_name=datastore_name,
            command=batch_create_command)

        if res.errors:
            error_messages = []
            for artifact_name in res.errors:
                error = res.errors[artifact_name].error
                error_messages.append("{}: {}".format(error.code,
                                                      error.message))
            raise AzureMLAggregatedException(error_messages)

        return res

    def upload_stream_to_existing_artifact(self, stream, artifact, content_information,
                                           content_type=None, session=None):
        """upload a stream to existring artifact"""
        artifact = artifact
        artifact_uri = content_information.content_uri
        session = session if session is not None else self.session
        res = upload_blob_from_stream(stream, artifact_uri, content_type=content_type, session=session,
                                      timeout=TIMEOUT, backoff=BACKOFF_START, retries=RETRY_LIMIT)
        return res

    def upload_artifact_from_stream(self, stream, origin, container, name, content_type=None,
                                    session=None, datastore_name=None):
        """upload a stream to a new artifact"""
        # Construct body
        res = self.create_empty_artifacts(origin, container, name, datastore_name)
        artifact = res.artifacts[name]
        content_information = res.artifact_content_information[name]
        self.upload_stream_to_existing_artifact(stream, artifact, content_information,
                                                content_type=content_type, session=session)
        return res

    def upload_artifact_from_path(self, path, *args, **kwargs):
        """upload a local file to a new artifact"""
        path = os.path.normpath(path)
        path = os.path.abspath(path)
        with open(path, "rb") as stream:
            return self.upload_artifact_from_stream(stream, *args, **kwargs)

    def upload_artifact(self, artifact, *args, **kwargs):
        """upload local file or stream to a new artifact"""
        self._logger.debug("Called upload_artifact")
        if isinstance(artifact, str):
            self._logger.debug("Uploading path artifact")
            return self.upload_artifact_from_path(artifact, *args, **kwargs)
        elif isinstance(artifact, IOBase):
            self._logger.debug("Uploading io artifact")
            return self.upload_artifact_from_stream(artifact, *args, **kwargs)
        else:
            raise UserErrorException("UnsupportedType: type {} is invalid, "
                                     "supported input types: file path or file".format(type(artifact)))

    def upload_files(self, paths, origin, container, names=None, return_artifacts=False,
                     timeout_seconds=None, datastore_name=None):
        """
        upload files to artifact service
        :rtype: list[BatchArtifactContentInformationDto]
         """

        if container is None:
            raise UserErrorException("Data Container ID cannot be null when uploading artifact")

        if timeout_seconds is None:
            timeout_seconds = int(
                os.environ.get(AZUREML_ARTIFACTS_TIMEOUT_ENV_VAR, AZUREML_ARTIFACTS_MIN_TIMEOUT))

        names = names if names is not None else paths
        path_to_name = {}
        paths_and_names = []
        # Check for duplicates, this removes possible interdependencies
        # during parallel uploads
        for path, name in zip(names, paths):
            if path not in path_to_name:
                paths_and_names.append((path, name))
                path_to_name[path] = name
            else:
                self._logger.warning("Found repeat file {} with name {} in upload_files.\n"
                                     "Uploading file {} to the original name "
                                     "{}.".format(path, name, path, path_to_name[path]))

        batch_size = SUPPORTED_NUM_EMPTY_ARTIFACTS

        results = []
        artifacts = {}
        for i in range(0, len(names), batch_size):
            with TaskQueue(worker_pool=self._pool, flush_timeout_seconds=timeout_seconds,
                           _ident="upload_files", _parent_logger=self._logger) as task_queue:
                batch_names = names[i:i + batch_size]
                batch_paths = paths[i:i + batch_size]

                content_information = self.create_empty_artifacts(origin, container, batch_names, datastore_name)
                artifacts.update(content_information.artifacts)

                def perform_upload(path, artifact, artifact_content_info, session):
                    with open(path, "rb") as stream:
                        return self.upload_stream_to_existing_artifact(stream, artifact, artifact_content_info,
                                                                       session=session)

                for path, name in zip(batch_paths, batch_names):
                    artifact = content_information.artifacts[name]
                    artifact_content_info = content_information.artifact_content_information[name]
                    task = task_queue.add(perform_upload, path, artifact, artifact_content_info, self.session)
                    results.append(task)

        if return_artifacts:
            return artifacts, map(lambda task: task.wait(), results)
        else:
            return map(lambda task: task.wait(), results)

    def upload_dir(self, dir_path, origin, container, path_to_name_fn=None, datastore_name=None):
        """
        upload all files in path
        :rtype: list[BatchArtifactContentInformationDto]
        """
        if not os.path.isdir(dir_path):
            raise UserErrorException("Cannot upload path: {} since it is not a valid directory.".format(dir_path))
        paths_to_upload = []
        names = []
        for pathl, _subdirs, files in os.walk(dir_path):
            for _file in files:
                fpath = os.path.join(pathl, _file)
                paths_to_upload.append(fpath)
                if path_to_name_fn is not None:
                    name = path_to_name_fn(fpath)
                else:
                    name = fpath
                names.append(name)
        self._logger.debug("Uploading {}".format(names))
        result = self.upload_files(paths_to_upload, origin, container, names, datastore_name=datastore_name)
        return result

    def get_file_uri(self, origin, container, path, session=None):
        """get the readable sas uri of an artifact"""
        res = self._execute_with_workspace_arguments(self._client.artifact.get_content_information,
                                                     origin=origin,
                                                     container=container,
                                                     path=path)
        return res.content_uri

    def get_file_by_artifact_id(self, artifact_id):
        """
        get sas uri of an artifact
        """
        # TODO change name to get file uri from artifact id
        # get SAS using get_content_info if id, else list sas by prefix
        # download sas to path
        [origin, container, path] = artifact_id.split("/", 2)
        local_path = os.path.abspath(path)
        return (local_path, self.get_file_uri(origin, container, path))

    def get_files_by_artifact_prefix_id(self, artifact_prefix_id):
        """
        get sas urls under a prefix artifact id
        """

        files_to_download = []
        # get SAS using get_content_info if id, else list sas by prefix
        # download sas to path
        [origin, container, prefix] = artifact_prefix_id.split("/", 2)
        self._logger.debug("Fetching files for prefix in {}, {}, {}".format(origin, container, prefix))
        dtos = self._execute_with_workspace_arguments(self._client.artifact.list_sas_by_prefix,
                                                      origin=origin,
                                                      container=container,
                                                      path=prefix,
                                                      is_paginated=True)
        for dto in dtos:
            path = dto.path
            local_path = path
            sas_uri = dto.content_uri
            files_to_download.append((local_path, sas_uri))
        return files_to_download

    def download_artifact(self, origin, container, path, output_file_path, _validate_checksum=False):
        """
        Download a single artifact from artifact service

        :param origin: the high-level origin of the artifact
        :type origin: str
        :param container: the container within the origin
        :type container: str
        :param path: the filepath within the container of the artifact to be downloaded
        :type path: str
        :param output_file_path: filepath in which to store the downloaded artifact locally
        :rtype: None
        """
        filename = os.path.basename(path)  # save outputs/filename.txt as filename.txt
        if os.path.isdir(output_file_path):
            self._logger.debug("output_file_path for download_artifact is a directory.")
            output_file_path = os.path.join(output_file_path, filename)
        else:
            self._logger.debug("output_file_path for download_artifact is not a directory.")

        try:
            content_info = self._execute_with_workspace_arguments(self._client.artifact.get_content_information,
                                                                  origin, container, path)
            if not content_info:
                raise UserErrorException("Cannot find the artifact '{0}' in container '{1}'".format(path, container))
            uri = content_info.content_uri
            self._execute_func(download_file, uri, output_file_path, session=self.session,
                               _validate_check_sum=_validate_checksum)
        except HttpOperationError as operation_error:
            self._handle_http_operation_error(operation_error, origin, container, path)

    def download_artifacts_from_prefix(
            self, origin, container, prefix=None, output_directory=None, output_paths=None,
            batch_size=100, append_prefix=True, fail_on_not_found=False, timeout_seconds=None):
        """
        Download multiple files from artifact service under a given container or prefix if specified

        :param origin: the high-level origin of the artifact
        :type origin: str
        :param container: the container within the origin
        :type container: str
        :param prefix: the filepath prefix within the container from which to download all artifacts
        :type prefix: str
        :param output_directory: optional directory that all artifact paths use as a prefix
        :type output_directory: str
        :param output_paths: optional filepaths in which to store the downloaded artifacts.
            Should be unique and match length of paths.
        :type output_paths: [str]
        :param batch_size: number of files to download per batch
        :type batch_size: int
        :param append_prefix: optional flag to append the specified prefix from the final output file path
            if False then the the prefix is removed from the output file path. If output_paths are specified
            however then it does not remove the prefix as it would be removing it from the user specified
            output_paths.
        :type append_prefix: bool
        :param fail_on_not_found: optional flag to raise exception if BlobNotFound instead of retrying
        :type fail_on_not_found: bool
        :param timeout_seconds: The timeout for downloading files.
        :type timeout_seconds: int
        :rtype: None
        """
        container_path = "{}/{}/".format(origin, container)
        if not prefix:
            prefix = ''
        else:
            prefix = prefix.replace("\\", "/")
            if (prefix[-1] != "/"):
                prefix += "/"
        prefix_path = container_path if len(prefix) == 0 else "{}{}/".format(container_path, prefix)

        try:
            sas_urls = self._execute_func(self.get_files_by_artifact_prefix_id, prefix_path)
        except HttpOperationError as operation_error:
            self._handle_http_operation_error(operation_error, origin, container, prefix, prefix=True)

        output_path_specified = output_paths is not None
        paths = [url_tuple[0] for url_tuple in sas_urls]
        sas_urls = list(map(lambda x: x[1], sas_urls))
        output_paths = output_paths if output_paths is not None else paths
        if len(output_paths) != len(paths) or len(set(output_paths)) != len(output_paths):
            self._logger.warning("Length of output paths is not the same as the length of paths"
                                 "or output_paths contains duplicates. Using paths as output_paths.")
            output_paths = paths

        if output_directory is not None:
            if not append_prefix and not output_path_specified:
                output_paths = list(map(lambda x: x.replace(prefix, "", 1), output_paths))
                self._logger.debug("Set of output paths without appended prefix: {}".format(output_paths))
            output_paths = list(map(lambda x: os.path.join(output_directory, x), output_paths))
            self._logger.debug("Final set of calculated output paths: {}".format(output_paths))

        for i in range(0, len(sas_urls), batch_size):
            try:
                with TaskQueue(worker_pool=self._pool,
                               _ident="download_files",
                               _parent_logger=self._logger,
                               flush_timeout_seconds=timeout_seconds) as task_queue:
                    batch_urls = sas_urls[i:i + batch_size]
                    batch_output_paths = output_paths[i:i + batch_size]

                    def perform_download_file(url, output_file_path, session):
                        self._execute_func(
                            download_file, url, output_file_path, session=session, fail_on_not_found=fail_on_not_found
                        )

                    for url, output_path in zip(batch_urls, batch_output_paths):
                        task_queue.add(perform_download_file, url, output_path, self.session)
            except AzureMLException as error:
                self._logger.debug("Download of artifact batch {} failed with error: {}"
                                   .format(str(sas_urls[i: i + batch_size]), error))
                raise

    def download_artifact_contents_to_string(self, origin, container, path, encoding="utf-8"):
        """download metric stored as a json artifact"""
        uri = self.get_artifact_uri(origin, container, path)
        return download_file_stream(source_uri=uri, session=self.session, encoding=encoding)

    def download_artifact_contents_to_bytes(self, origin, container, path):
        uri = self.get_artifact_uri(origin, container, path)
        return download_file_stream(source_uri=uri, download_to_bytes=True, session=self.session)

    def get_file_paths(self, origin, container):
        """list artifact info"""
        artifacts = self._execute_with_workspace_arguments(self._client.artifact.list_in_container,
                                                           origin=origin, container=container,
                                                           is_paginated=True)

        return map(lambda artifact_dto: artifact_dto.path, artifacts)

    def batch_ingest_from_sas(self, origin, container, container_sas,
                              container_uri, prefix, artifact_prefix):
        """get artifact info"""

        command = BatchArtifactContainerSasIngestCommand(container_sas,
                                                         container_uri,
                                                         prefix,
                                                         artifact_prefix)
        res = self._execute_with_workspace_arguments(self._client.artifact.batch_ingest_from_sas,
                                                     origin, container, command)
        return res

    def get_artifact_by_container(self, origin, container):
        """
        Get artifact names of a run by their run_id
        :param origin: origin component of the artifactId
        :type origin: str
        :param container: container component of the artifactId
        :type container: str
        :return: a generator of ~_restclient.models.ArtifactDto
        """
        return self._execute_with_workspace_arguments(self._client.artifact.list_in_container,
                                                      origin=origin,
                                                      container=container,
                                                      is_paginated=True)

    def get_artifact_uri(self, origin, container, attachment_name, is_async=False):
        """
        Get the uri of artifact being saved of a run by its run_id and name
        :param origin: origin component of the artifactId
        :type origin: str
        :param container: container component of the artifactId
        :type container: str
        :param attachment_name: path component of the artifactId
        :type attachment_name: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async.AsyncTask object
            If parameter is_async is False or missing,
            return str
        """
        try:
            content_info = self._execute_with_workspace_arguments(self._client.artifact.get_content_information,
                                                                  origin=origin,
                                                                  container=container,
                                                                  path=attachment_name,
                                                                  is_async=is_async)
            if not content_info:
                raise UserErrorException("Cannot find the artifact '{0}' in container '{1}'".format(attachment_name,
                                                                                                    container))
            uri = content_info.content_uri
            return uri
        except HttpOperationError as operation_error:
            self._handle_http_operation_error(operation_error, origin, container, attachment_name)

    def peek_artifact_content(self, source_uri):
        """
        Get the text of an artifact
        :param source_uri:  Source URI to read the artifact from
        :type source_uri: str
        :return: Text of the source artifact
        """
        res = download_file_stream(
            source_uri, max_retries=self.retries, stream=False, session=self.session)
        return res

    def upload_artifact_content(self, origin, container, attachment_name, content=None,
                                index=None, append=None, is_async=False):
        """
        Upload content to artifact of a run.
        :param origin: origin component of the artifactId
        :type origin: str
        :param container: container component of the artifactId
        :type container: str
        :param attachment_name: path component of the artifactId
        :type attachment_name: str
        :param content: content to upload
        :param index: block index in artifact to write to (optional)
        :type index: int
        :param append: if true, content is appended to artifact (optional)
        :type append: bool
        :param is_async: execute request asynchronously (optional)
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async.AsyncTask object
            If parameter is_async is False or missing,
            return: ~_restclient.models.ArtifactDto
        """
        return self._execute_with_workspace_arguments(self._client.artifact.upload,
                                                      origin=origin,
                                                      container=container,
                                                      path=attachment_name,
                                                      content=content,
                                                      index=index,
                                                      append=append,
                                                      is_async=is_async)

    def get_writeable_artifact_sas_uri(self, origin, container, path, is_async=False):
        """
        Get a writeable sas uri of artifact with id aml://artifactId/{origin}/{container}/{path}
        :param origin: origin component of the artifactId
        :type origin: str
        :param container: container component of the artifactId
        :type container: str
        :param path: path component of the artifactId
        :type path: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async.AsyncTask object
            If parameter is_async is False or missing,
            return str
        """
        try:
            content_info = self._execute_with_workspace_arguments(self._client.artifact.get_write_sas,
                                                                  origin=origin,
                                                                  container=container,
                                                                  path=path,
                                                                  is_async=is_async)
            if not content_info:
                raise UserErrorException("Cannot find the artifact '{0}' in container '{1}'".format(path,
                                                                                                    container))
            uri = content_info.content_uri
            return uri
        except HttpOperationError as operation_error:
            self._handle_http_operation_error(operation_error, origin, container, path)

    def _handle_http_operation_error(self, operation_error, origin, container, path, prefix=False):
        """
        Handles HttpOperationError received from Artifact Service
        :param operation_error: the error received
        :type operation_error: HttpOperationError
        :param origin: origin component of the artifactId
        :type origin: str
        :param container: container component of the artifactId
        :type container: str
        :param path: path component of the artifactId
        :type path: str
        :param prefix: boolean, true if the path represents a directory, false if a single file
        :type prefix: bool
        """
        if operation_error.response.status_code == 404:
            existing_files = self.get_file_paths(origin, container)
            type_string = "Prefix" if prefix else "File"
            raise UserErrorException("{0} with path {1} was not found,\n"
                                     "available files include: "
                                     "{2}.".format(type_string, path, ",".join(existing_files)))
        else:
            raise operation_error
