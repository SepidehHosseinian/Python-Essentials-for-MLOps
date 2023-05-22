# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import multiprocessing

from threading import Thread, Event

from azureml._file_utils import get_block_blob_service_credentials

from azureml._common.async_utils.task_queue import TaskQueue

from azureml._vendor.azure_storage.blob import BlobClient


AZUREML_LOG_FILE_SUFFIX = "azureml.log"


class AwaitableUpload(object):
    def __init__(self, fullpath, files_watched, upload_info, logger, artifacts_client,
                 origin, container, artifact_path):
        self.fullpath = fullpath
        self.artifact_path = artifact_path
        self.files_watched = files_watched
        self.upload_info = upload_info
        self.logger = logger.getChild("AwaitableUpload")
        self.artifacts_client = artifacts_client
        self.origin = origin
        self.container = container

    def result(self):
        try:
            self._upload()
        except Exception as ex:
            self.logger.warning("Failed to update blob with error:\n{}".format(ex))
            self.logger.warning("Refreshing SAS and trying again")
            # assume exception was caused by expired sas and refresh
            self._refresh_sas_token_and_upload()

    def _refresh_sas_token_and_upload(self):
        artifact_uri = self.artifacts_client.get_writeable_artifact_sas_uri(
            self.origin, self.container, self.artifact_path)
        segments = get_block_blob_service_credentials(artifact_uri)
        sas_token, account_name, endpoint_suffix, container_name, blob_name = segments
        account_url = "https://{account_name}.blob.{endpoint}".format(
            account_name=account_name, endpoint=endpoint_suffix
        )
        blob_service = BlobClient(
            account_url=account_url,
            credential=sas_token,
            container_name=container_name,
            blob_name=blob_name
        )
        self.upload_info[self.fullpath] = (blob_service, container_name, blob_name)
        try:
            self._upload()
        except Exception as ex:
            self.logger.warning("Failed to update blob with error:\n{}".format(ex))

    def _upload(self):
        last_uploaded_byte = self.files_watched[self.fullpath]
        (blob_service, container_name, blob_name) = self.upload_info[self.fullpath]
        chunk_size_bytes = os.environ.get("AZUREML_UPLOAD_CHUNK_SIZE_BYTES", 4 * 1000 * 1024)
        with open(self.fullpath, 'rb') as data:
            data.seek(last_uploaded_byte)
            read_bytes = data.read(chunk_size_bytes)
            if not read_bytes:
                return
            blob_service.upload_blob(read_bytes, blob_type="AppendBlob")
            last_uploaded_byte += len(read_bytes)
            self.files_watched[self.fullpath] = last_uploaded_byte


class UploadTask():
    def __init__(self, fullpath, files_watched, upload_info, files_uploading, logger,
                 artifacts_client, origin, container, artifact_path):
        self.fullpath = fullpath
        self.upload = AwaitableUpload(fullpath, files_watched, upload_info, logger,
                                      artifacts_client, origin, container, artifact_path)
        self.files_uploading = files_uploading

    def wait(self):
        self.upload.result()
        self.files_uploading.remove(self.fullpath)

    def result(self):
        return self.wait()


class FileWatcher(Thread):
    def __init__(self, directories_to_watch, origin, container, artifacts_client, logger,
                 parallelism=None, azureml_log_file_path=None):
        # Note: tried to use the Daemon base class, but it
        # doesn't provide the event mechanism to break away
        # from sleep earlier than the specified interval when
        # finish is called
        super(FileWatcher, self).__init__(daemon=True)
        self.directories_to_watch = {os.path.abspath(directory): os.path.normpath(directory)
                                     for directory in directories_to_watch}
        self.origin = origin
        self.container = container
        self.artifacts_client = artifacts_client
        self.logger = logger.getChild("FileWatcher")
        self._event = Event()
        if parallelism is None:
            parallelism = multiprocessing.cpu_count()
        self.parallelism = parallelism
        if azureml_log_file_path is not None:
            azureml_log_file_path = os.path.normpath(azureml_log_file_path)
        self.azureml_log_file_path = azureml_log_file_path
        azureml_log_abspath = os.path.abspath(azureml_log_file_path) if azureml_log_file_path is not None else None
        self.azureml_log_file_abspath = azureml_log_abspath

    def create_artifacts(self, new_files):
        try:
            posix_artifact_id_paths = [os.path.normpath(artifact_id_path).replace(os.sep, '/')
                                       for (leaf_file, fullpath, artifact_id_path) in new_files]
            local_abspaths = [fullpath for (leaf_file, fullpath, artifact_id_path) in new_files]
            posix_to_local = dict(zip(posix_artifact_id_paths, local_abspaths))
            # Create artifact to get sas URL
            res = self.artifacts_client.create_empty_artifacts(self.origin,
                                                               self.container,
                                                               posix_artifact_id_paths)

            artifact_keys = list(res.artifacts.keys())
            artifacts = [res.artifacts[artifact_name] for artifact_name in artifact_keys]
            returned_paths = [posix_to_local[artifact.path] for artifact in artifacts]
            artifact_uris = [res.artifact_content_information[name].content_uri for name in artifact_keys]
            return (artifact_uris, returned_paths, True)
        except Exception as ex:
            self.logger.debug("Exception creating artifacts:\n{}".format(ex))
            return ([], [], False)

    def create_blobs(self, artifact_uris, files_watched, upload_info, full_paths):
        try:
            for artifact_uri, fullpath in zip(artifact_uris, full_paths):
                segments = get_block_blob_service_credentials(artifact_uri)
                sas_token, account_name, endpoint_suffix, container_name, blob_name = segments
                # Create the blob service and blob to upload file to
                account_url = "https://{account_name}.blob.{endpoint}".format(
                    account_name=account_name, endpoint=endpoint_suffix
                )
                blob_service = BlobClient(
                    account_url=account_url,
                    credential=sas_token,
                    container_name=container_name,
                    blob_name=blob_name
                )
                blob_service.create_append_blob()
                self.logger.debug("uploading data to container: {} blob: {} path: {}".format(container_name,
                                                                                             blob_name,
                                                                                             fullpath))
                files_watched[fullpath] = 0
                upload_info[fullpath] = (blob_service, container_name, blob_name)
            return True
        except Exception as ex:
            self.logger.debug("Exception creating blobs:\n{}".format(ex))
            return False

    def walk_files(self, files_watched, files_uploading, upload_info, tq, current_stat):
        leaf_files = {}
        for abspath_directory, directory in self.directories_to_watch.items():
            if (not os.path.isdir(abspath_directory)) and os.path.isfile(abspath_directory):
                artifact_id_path = os.path.normpath(directory)
                abspath = abspath_directory
                if leaf_files.get(abspath) is None:
                    if (AZUREML_LOG_FILE_SUFFIX in abspath and abspath != self.azureml_log_file_abspath):
                        continue
                    leaf_file = os.path.basename(abspath)
                    leaf_files[abspath] = (leaf_file, abspath, artifact_id_path)
                continue
            artifact_id_basepath = directory
            abspath_directory_length = len(abspath_directory)
            for root, dirs, files in os.walk(abspath_directory):
                artifact_id_dir_path = os.path.normpath(artifact_id_basepath) + root[abspath_directory_length:]
                artifact_id_dir_path = os.path.normpath(artifact_id_dir_path)
                for leaf_file in files:
                    fullpath = os.path.join(root, leaf_file)
                    artifact_id_path = os.path.join(artifact_id_dir_path, leaf_file)
                    abspath = os.path.abspath(fullpath)
                    if leaf_files.get(abspath) is None:
                        if (AZUREML_LOG_FILE_SUFFIX in abspath and abspath != self.azureml_log_file_abspath):
                            continue
                        leaf_files[abspath] = (leaf_file, fullpath, artifact_id_path)
        # Ignore hidden files
        visible_files = [(leaf_file, fullpath, artifact_id_path) for (leaf_file, fullpath, artifact_id_path)
                         in leaf_files.values() if not leaf_file.startswith(".")]
        new_files = [(leaf_file, fullpath, artifact_id_path) for (leaf_file, fullpath, artifact_id_path)
                     in leaf_files.values() if fullpath not in files_watched]
        if new_files:
            # For all new files do a batch create
            (artifact_uris, full_paths, success) = self.create_artifacts(new_files)
            # Exit on error, stop watcher
            if not success:
                self.logger.debug("Exiting File Watcher due to errors with creating artifacts\n")
                return
            # For new files create all blobs in storage
            success = self.create_blobs(artifact_uris, files_watched, upload_info, full_paths)
            # Exit on error, stop watcher
            if not success:
                self.logger.debug("Exiting File Watcher due to errors with creating blobs\n")
                return

        try:
            # Iterate over all files and create a task to upload a chunk of data
            for (leaf_file, fullpath, artifact_id_path) in visible_files:
                # Check if file size has changed, ignore files that haven't changed
                filesize = os.stat(fullpath).st_size
                if files_watched[fullpath] >= filesize:
                    continue
                current_stat[fullpath] = filesize
                if fullpath not in files_uploading:
                    files_uploading.add(fullpath)
                    # start an async task to upload the file
                    upload_task = UploadTask(fullpath, files_watched, upload_info, files_uploading,
                                             self.logger, self.artifacts_client, self.origin,
                                             self.container,
                                             os.path.normpath(artifact_id_path).replace(os.sep, '/'))
                    tq.add(upload_task.result)

        except Exception as ex:
            self.logger.debug("Exiting File Watcher due to errors creating upload tasks:\n{}".format(ex))
            return
        sleep_interval_sec = 10
        self._event.wait(timeout=sleep_interval_sec)

    def uploaded_to_stat(self, files_watched, last_stat):
        for file_watched in files_watched:
            if file_watched in last_stat and last_stat[file_watched] > files_watched[file_watched]:
                return False
        return True

    def run(self):
        files_watched = {}
        # Take a stat of the current directory
        current_stat = {}
        files_uploading = set()
        upload_info = {}
        with TaskQueue(_ident="UploadQueue", _parent_logger=self.logger) as tq:
            while not self._event.is_set():
                self.walk_files(files_watched, files_uploading, upload_info, tq, current_stat)
            self.logger.debug("FileWatcher received exit event, getting current_stat")
            # Walk until files have uploaded at least to last stat
            # If more data has been uploaded since the last stat from some other process we won't continue to upload
            self.walk_files(files_watched, files_uploading, upload_info, tq, current_stat)
            self.logger.debug("FileWatcher retrieved current_stat, will upload to current_stat")
            dummy_stat = {}
            while (not self.uploaded_to_stat(files_watched, current_stat)):
                self.logger.debug("FileWatcher uploading files to current_stat...")
                self.walk_files(files_watched, files_uploading, upload_info, tq, dummy_stat)
            self.logger.debug("FileWatcher finished uploading to current_stat, finishing task queue")
            # Finish the task queue

        # Do a final pass at uploading azureml.log
        # This is necessary because once the signal to terminate FileWatcher is received
        # FileWatcher does a final pass at all the files. If azureml.log is uploaded
        # before any of the other files, there can be new log statements to azureml.log
        # belonging to the other files, so we have to do a final pass at azureml.log
        if self.azureml_log_file_path is not None:
            self._final_upload_azureml_log(files_watched, upload_info)

    def finish(self):
        self.logger.debug("FileWatcher called finish, setting event")
        self._event.set()

    def _final_upload_azureml_log(self, files_watched, upload_info):
        task = UploadTask(self.azureml_log_file_abspath, files_watched, upload_info, {self.azureml_log_file_abspath},
                          self.logger, self.artifacts_client, self.origin, self.container,
                          os.path.normpath(self.azureml_log_file_path).replace(os.sep, "/"))
        task.wait()
