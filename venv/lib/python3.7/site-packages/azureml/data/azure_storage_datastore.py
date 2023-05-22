# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for datastores that save connection information to Azure Blob and Azure File storage."""
import logging
import os
import re
import requests
import json
import collections

from abc import ABCMeta, abstractmethod
from itertools import tee
from azure.common import AzureHttpError
from azureml._vendor.azure_storage.blob import BlobServiceClient
from azureml._vendor.azure_storage.fileshare import ShareDirectoryClient, ShareFileClient, FileProperties
from azureml._vendor.azure_storage.fileshare import ShareClient

from azureml._base_sdk_common.utils import create_retry, common_path, to_unix_path, accumulate
from azureml._common.async_utils.async_task import AsyncTask
from azureml._common.async_utils.task_queue import TaskQueue
from azureml.data._dataset_deprecation import deprecated
from azureml.data.abstract_datastore import AbstractDatastore
from azureml.data.data_reference import DataReference
from azureml.exceptions import UserErrorException

module_logger = logging.getLogger(__name__)


FOLDER = 0
EMPTY_FILE = 1
FILE = 2


class AbstractAzureStorageDatastore(AbstractDatastore):
    """Represents the base class for datastores that save connection information to Azure Blob and Azure File storage.

    You should not work with this class directly. To create a datastore, use one of the ``register*`` methods
    of the Datastore class, for example, :meth:`azureml.core.datastore.Datastore.register_azure_blob_container`.

    Note: When using a datastore to access data, you must have permission to access the data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: azureml.core.workspace.Workspace
    :param name: The name of the datastore. It can only contain alphanumeric
        characters or - or _.
    :type name: str
    :param datastore_type: The type of this datastore, either "AzureBlob" or "AzureFile".
    :type datastore_type: str
    :param container_name: The container name.
    :type container_name: str
    :param account_name: The storage account name.
    :type account_name: str
    :param sas_token: The SAS token for accessing this container, defaults to None.
    :type sas_token: str, optional
    :param account_key: The storage account key, defaults to None.
    :type account_key: str, optional
    :param protocol: The protocol to use to connect to the storage account.
        If None, defaults to https.
    :type protocol: str, optional
    :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
    :type endpoint: str, optional
    """

    __metaclass__ = ABCMeta

    _sanitize_regex = re.compile(r"^(\.*[/\\])*")

    def __init__(self, workspace, name, datastore_type, container_name, account_name,
                 sas_token=None, account_key=None, protocol=None, endpoint=None):
        """Class AbstractAzureStorageDatastore constructor.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the datastore. It can only contain alphanumeric
            characters or - or _.
        :type name: str
        :param datastore_type: The type of this datastore, either "AzureBlob" or "AzureFile".
        :type datastore_type: str
        :param container_name: The container name.
        :type container_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param sas_token: The SAS token for accessing this container, defaults to None.
        :type sas_token: str, optional
        :param account_key: The storage account key, defaults to None.
        :type account_key: str, optional
        :param protocol: The protocol to use to connect to the storage account.
            If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        """
        super(AbstractAzureStorageDatastore, self).__init__(workspace, name, datastore_type)
        self.container_name = container_name
        self.account_name = account_name
        self.sas_token = sas_token
        self.account_key = account_key
        self.credential_type = 'None'
        self.protocol = protocol
        self.endpoint = endpoint

        if account_key:
            self.credential_type = 'AccountKey'
        if sas_token:
            self.credential_type = 'Sas'

        self._num_workers = 32

        self._data_reference = DataReference(datastore=self)

    def __repr__(self):
        """Return the string representation of the AbstractAzureStorageDatastore object.

        :return: String representation of the AbstractAzureStorageDatastore object
        :rtype: str
        """
        content = collections.OrderedDict()

        content['name'] = self.name
        content['container_name'] = self.container_name
        content['account_name'] = self.account_name
        content['protocol'] = self.protocol
        content['endpoint'] = self.endpoint

        return json.dumps(content, indent=2)

    def __str__(self):
        """Return the string representation of the AbstractAzureStorageDatastore object.

        :return: String representation of the AbstractAzureStorageDatastore object
        :rtype: str
        """
        return self.__repr__()

    def path(self, path=None, data_reference_name=None):
        """Return corresponding data reference object.

        :param path: The relative path on the datastore.
        :type path: str
        :param data_reference_name: The name of the data reference.
        :type data_reference_name: str
        :return: The data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        return self._data_reference.path(path, data_reference_name)

    @deprecated(
        '"datastore.as_download"',
        '"file_dataset.as_download" instead')
    def as_download(self, path_on_compute=None):
        """Return data reference object with download mode.

        Note: This method is deprecated and will no longer be supported. Recommend to use
        ``file_dataset.as_download`` instead.

        :param path_on_compute: The relative path on the compute.
        :type path_on_compute: str
        :return: The data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        return self._data_reference.as_download(path_on_compute)

    @deprecated(
        '"datastore.as_upload"',
        '"dataset" to update instead')
    def as_upload(self, path_on_compute=None):
        """Return data reference object with upload mode.

        Note: This method is deprecated and will no longer be supported. Recommend to use
        dataset to upload instead.

        :param path_on_compute: The relative path on the compute.
        :type path_on_compute: str
        :return: The data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        return self._data_reference.as_upload(path_on_compute)

    @deprecated(
        '"datastore.as_mount"',
        '"file_dataset.as_mount" instead')
    def as_mount(self):
        """Return data reference object with mount mode.

        Note: This method is deprecated and will no longer be supported. Recommend to use
        ``file_dataset.as_mount`` instead.

        :param path_on_compute: The relative path on the compute.
        :type path_on_compute: str
        :return: The data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        return self._data_reference.as_mount()

    @abstractmethod
    def download(self, target_path, prefix=None, overwrite=False, show_progress=True):
        """Download paths with prefix to target_path.

        :param target_path:
        :param prefix:
        :param overwrite:
        :param show_progress:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def upload(self, src_dir, target_path=None, overwrite=False, show_progress=True):
        """Upload src_dir to target_path.

        :param src_dir:
        :param target_path:
        :param overwrite:
        :param show_progress:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def upload_files(self, files, relative_root=None, target_path=None, overwrite=False, show_progress=True):
        """Upload files to target_path.

        :param files:
        :param relative_root: relative path in target
        :param target_path:
        :param overwrite:
        :param show_progress:
        :return:
        """
        raise NotImplementedError()

    def _get_data_reference(self):
        return self._data_reference

    @property
    def is_sas(self):
        """Deprecated: use ``credential_type`` property.

        This property is deprecated, please use the property ``credential_type`` to determine the credential type.
        """
        module_logger.warning("This property is deprecated, please use the property \"credential_type\""
                              + " to determine the credential type.")
        return bool(self.sas_token)

    def _as_dict(self, hide_secret=True):
        output = super(AbstractAzureStorageDatastore, self)._as_dict()
        output["container_name"] = self.container_name
        output["account_name"] = self.account_name
        output["protocol"] = self.protocol
        output["endpoint"] = self.endpoint

        if not hide_secret:
            output["credential_type"] = self.credential_type
            output["sas_token"] = self.sas_token
            output["account_key"] = self.account_key

        return output

    def _get_default_request_session(self):
        a = requests.adapters.HTTPAdapter(pool_connections=self._num_workers, pool_maxsize=self._num_workers * 2,
                                          max_retries=create_retry())
        s = requests.Session()
        s.mount("http://", a)
        s.mount("https://", a)
        return s

    def _get_upload_from_dir(self, src_path, target_path):
        src_path = src_path.rstrip("/\\")
        if not os.path.isdir(src_path):
            raise UserErrorException("src_path must be a directory.")

        paths_to_upload = []
        for dirpath, dirnames, filenames in os.walk(src_path):
            paths_to_upload += self._get_upload_from_files(
                map(lambda f: os.path.join(dirpath, f), filenames),
                target_path,
                src_path,
                True)
        return paths_to_upload

    def _get_upload_from_files(self, file_paths, target_path, relative_root, skip_root_check):
        paths_to_upload = []
        target_path = AbstractAzureStorageDatastore._sanitize_target_path(target_path)
        for file_path in file_paths:
            if not skip_root_check and relative_root not in file_path and relative_root != "/":
                raise UserErrorException("relative_root: '{}' is not part of the file_path: '{}'.".format(
                    relative_root, file_path))
            if not os.path.isfile(file_path):
                err_msg = "'{}' does not point to a file. " + \
                    "Please upload the file to cloud first if running in a cloud notebook."
                raise UserErrorException(err_msg.format(file_path))

            target_file_path = to_unix_path(file_path)
            if relative_root != "/":
                # need to do this because Windows doesn't support relpath if the partition is different
                target_file_path = os.path.relpath(target_file_path, to_unix_path(relative_root))
            else:
                # strip away / otherwise we will create a folder in the container with no name
                target_file_path = target_file_path.lstrip("/")

            if target_path:
                target_file_path = os.path.join(target_path, target_file_path)

            paths_to_upload.append((file_path, target_file_path))

        return paths_to_upload

    @staticmethod
    def _sanitize_target_path(target_path):
        if not target_path:
            return target_path
        return AbstractAzureStorageDatastore._sanitize_regex.sub("", target_path)

    def _start_upload_task(self, paths_to_upload, overwrite, exists, show_progress, task_generator):
        # it's an estimated total because we might skip some files
        estimated_total = len(paths_to_upload)
        counter = _Counter()
        console = self._get_progress_logger(show_progress, module_logger)

        console("Uploading an estimated of {} files".format(estimated_total))

        def exception_handler(e, logger):
            logger.error("Upload failed, please make sure target_path does not start with invalid characters.", e)

        with TaskQueue(flush_timeout_seconds=float('inf'), _ident=__name__, _parent_logger=module_logger) as tq:
            for (src_file_path, target_file_path) in paths_to_upload:
                if not overwrite:
                    if exists(target_file_path):
                        estimated_total -= 1
                        console("Target already exists. Skipping upload for {}".format(target_file_path))
                        continue

                task_fn = task_generator(target_file_path, src_file_path)
                future_handler = self._get_task_handler(src_file_path, counter, estimated_total, show_progress,
                                                        "Upload", exception_handler)
                future = tq.create_future(task_fn)
                async_task = AsyncTask(future, handler=future_handler,
                                       _ident="task_upload_{}".format(target_file_path),
                                       _parent_logger=module_logger)
                tq.add_task(async_task)

        console("Uploaded {} files".format(counter.count()))
        return counter.count()

    def _get_task_handler(self, f, counter, total, show_progress, action, exception_handler=None):
        def handler(future, logger):
            print_progress = self._get_progress_logger(show_progress, logger)

            try:
                print_progress("{}ing {}".format(action, f))
                result = future.result()
                # thanks to GIL no need to use lock here
                counter.increment()
                print_progress("{}ed {}, {} files out of an estimated total of {}".format(
                    action, f, counter.count(), total))
                return result
            except Exception as e:
                if exception_handler:
                    exception_handler(e, logger)
                else:
                    logger.error("Task Exception", e)

        return handler

    def _get_progress_logger(self, show_progress, logger=None):
        console = self._get_console_logger()

        def log(message):
            show_progress and console.write("{}\n".format(message))
            logger.info(message)

        return log


class AzureFileDatastore(AbstractAzureStorageDatastore):
    """Represents a datastore that saves connection information to Azure File storage.

    You should not work with this class directly. To create a datastore of this type, use
    the :meth:`azureml.core.datastore.Datastore.register_azure_file_share` method of Datastore.

    Note: When using a datastore to access data, you must have permission to access that data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: azureml.core.workspace.Workspace
    :param name: The name of the datastore. It can only contain alphanumeric
        characters or - or _.
    :type name: str
    :param container_name: The file share name
    :type container_name: str
    :param account_name: The storage account name
    :type account_name: str
    :param sas_token: The SAS token for accessing this container. The default is None.
    :type sas_token: str, optional
    :param account_key: The storage account key. The default is None.
    :type account_key: str, optional
    :param protocol: The protocol to use to connect to the storage account. If None, defaults to https.
    :type protocol: str, optional
    :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
    :type endpoint: str, optional
    :param request_session: The session object to use for http requests. The default is None.
    :type request_session: requests.sessions.Session, optional
    """

    def __init__(self, workspace, name, container_name, account_name, sas_token=None,
                 account_key=None, protocol=None, endpoint='core.windows.net', request_session=None):
        """Initialize a new Azure File Share Datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the datastore. It can only contain alphanumeric
            characters or - or _.
        :type name: str
        :param container_name: The file share name
        :type container_name: str
        :param account_name: The storage account name
        :type account_name: str
        :param sas_token: The SAS token for accessing this container. The default is None.
        :type sas_token: str, optional
        :param account_key: The storage account key. The default is None.
        :type account_key: str, optional
        :param protocol: The protocol to use to connect to the storage account. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param request_session: The session object to use for http requests. The default is None.
        :type request_session: requests.sessions.Session, optional
        """
        super(AzureFileDatastore, self).__init__(workspace, name, 'AzureFile', container_name, account_name,
                                                 sas_token, account_key, protocol, endpoint)
        account_url = "https://{account_name}.file.{endpoint}".format(
            account_name=self.account_name, endpoint=self.endpoint
        )
        credential = None
        if self.sas_token:
            credential = self.sas_token
        elif self.account_key:
            credential = {
                'account_name': self.account_name,
                'account_key': self.account_key
            }
        self.file_service = ShareClient(
            account_url=account_url,
            credential=credential,
            share_name=self.container_name
        )

    def download(self, target_path, prefix=None, overwrite=False, show_progress=True):
        """Download the data from the file share to the local file system.

        :param target_path: The local directory to download the file to.
        :type target_path: str
        :param prefix: The path to the folder in the file share to download. If set to None, Azure Machine Learning
            will download everything in the file share.
        :type prefix: str, optional
        :param overwrite: Indicates whether to overwrite existing file.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show the progress of download in the console.
        :type show_progress: bool, optional
        :return: The number of files successfully downloaded.
        :rtype: int
        """
        module_logger.info("Called AzureFileDatastore.download")
        AzureFileDatastore._verify_prefix(prefix)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        total = 0
        counter = _Counter()
        remaining_paths = []

        with TaskQueue(flush_timeout_seconds=float('inf'), _ident=__name__, _parent_logger=module_logger) as tq:
            for file_or_dir in self.file_service.list_directories_and_files(
                    name_starts_with=prefix):
                remaining_paths.append((file_or_dir, ""))
            total = len(remaining_paths)
            while len(remaining_paths) > 0:
                file_or_dir, parent_path = remaining_paths.pop()
                if isinstance(file_or_dir, FileProperties):
                    file = file_or_dir
                    if not self._download_file(tq, parent_path, file.name, target_path,
                                               overwrite, total, counter, show_progress):
                        total -= 1
                else:
                    directory = file_or_dir
                    path = os.path.join(parent_path, directory.name)
                    account_url = "https://{account_name}.file.{endpoint}".format(
                        account_name=self.account_name, endpoint=self.endpoint
                    )
                    credential = None
                    if self.sas_token:
                        credential = self.sas_token
                    elif self.account_key:
                        credential = {
                            'account_name': self.account_name,
                            'account_key': self.account_key
                        }
                    file_service = ShareDirectoryClient(
                        account_url=account_url,
                        credential=credential,
                        share_name=self.container_name,
                        directory_path=path
                    )
                    for file_or_dir in file_service.list_directories_and_files():
                        remaining_paths.append((file_or_dir, path))
                        total += 1
        module_logger.info("Finished AzureFileDatastore.download. Downloaded {} files.".format(counter.count()))
        return counter.count()

    @deprecated(
        '"Datastore.upload"',
        '"Dataset.File.upload_directory" to upload your files \
            from a local directory and create FileDataset in single method call')
    def upload(self, src_dir, target_path=None, overwrite=False, show_progress=True):
        """Upload the data from the local file system to the file share this datastore points to.

        Note: For this method to work, the datastore object must have either account key or SAS
        token property populated. If datastore object doesn't contain any form of credential, this
        method will fail.
        Moreover, this method has been deprecated, may be removed in future. We suggest that you use
        the ``Dataset.File.upload_directory`` method to upload your files from local directory and
        create FileDataset in a single method call.

        :param src_dir: The local directory to upload.
        :type src_dir: str
        :param target_path: The location in file share to upload to. If None then upload to root.
        :type target_path: str
        :param overwrite: Indicates whether to overwrite existing files.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show the progress of upload in the console.
        :type show_progress: bool, optional
        :return: The DataReference instance for the target path uploaded.
        :rtype: azureml.data.data_reference.DataReference
        """
        module_logger.info("Called AzureFileDatastore.upload")
        target_path = target_path or ""
        account_url = "https://{account_name}.file.{endpoint}".format(
            account_name=self.account_name, endpoint=self.endpoint
        )
        credential = None
        if self.sas_token:
            credential = self.sas_token
        elif self.account_key:
            credential = {
                'account_name': self.account_name,
                'account_key': self.account_key
            }

        def check_file(target_file_path):
            service = ShareFileClient(
                account_url=account_url,
                credential=credential,
                share_name=self.container_name,
                file_path=target_file_path
            )
            try:
                service.get_file_properties()
                return True
            except Exception:
                return False

        count = self._start_upload_task(
            self._get_upload_from_dir(src_dir, target_path),
            overwrite,
            lambda target_file_path: check_file(target_file_path),
            show_progress,
            self._file_share_upload
        )
        module_logger.info("Finished AzureFileDatastore.upload with count={0}.".format(count))
        return DataReference(datastore=self, path_on_datastore=target_path)

    @deprecated(
        '"datastore.upload_files"',
        '"FileDatasetFactory.upload_directory" instead')
    def upload_files(self, files, relative_root=None, target_path=None, overwrite=False, show_progress=True):
        """Upload the data from the local file system to the file share this datastore points to.

        Note: This method is deprecated and will no longer be supported. Recommend to use
        ``FileDatasetFactory.upload_directory`` instead.

        :param files: A list of absolute path to files to upload.
        :type files: builtin.list[str]
        :param relative_root: The base path used to determine the path
            of the files in the file share. For example, if you upload /path/to/file.txt, and you define
            base path to be /path, when file.txt is uploaded to the file share, it will have
            the path of /to/file.txt. If ``target_path`` is also specified, then it will be used as
            the prefix for the derived path from above. The base path must be a common path of
            all of the files, otherwise an exception will be thrown. The default is None, which will find
            the common path.
        :type relative_root: str, optional
        :param target_path: The location in the file share to upload the data to. The default is None, the root.
        :type target_path: str, optional
        :param overwrite: Indicates whether to overwrite existing files.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show the progress of upload in the console.
        :type show_progress: bool, optional
        :return: The DataReference instance for the target path uploaded.
        :rtype: azureml.data.data_reference.DataReference
        """
        module_logger.info("Called AzureFileDatastore.upload_files")
        target_path = target_path or ""
        account_url = "https://{account_name}.file.{endpoint}".format(
            account_name=self.account_name, endpoint=self.endpoint
        )
        credential = None
        if self.sas_token:
            credential = self.sas_token
        elif self.account_key:
            credential = {
                'account_name': self.account_name,
                'account_key': self.account_key
            }

        def check_file(target_file_path):
            service = ShareFileClient(
                account_url=account_url,
                credential=credential,
                share_name=self.container_name,
                file_path=target_file_path
            )
            try:
                service.get_file_properties()
                return True
            except Exception:
                return False

        relative_root = relative_root or common_path(files)
        count = self._start_upload_task(
            self._get_upload_from_files(files, target_path, relative_root, False),
            overwrite,
            lambda target_file_path: check_file(target_file_path),
            show_progress,
            self._file_share_upload
        )
        module_logger.info("Finished AzureFileDatastore.upload_files with count={0}.".format(count))
        return DataReference(datastore=self, path_on_datastore=target_path)

    def _download_file(self, tq, dirpath, filename, target_path, overwrite, total, counter, show_progress):
        file_path = os.path.join(target_path, dirpath, filename)
        if not overwrite and os.path.exists(file_path):
            module_logger.warning("Path already exists. Skipping download for {}".format(file_path))
            return False

        if not os.path.exists(os.path.join(target_path, dirpath)):
            os.makedirs(os.path.join(target_path, dirpath))
        src_file_path = os.path.join(dirpath, filename)

        account_url = "https://{account_name}.file.{endpoint}".format(
            account_name=self.account_name, endpoint=self.endpoint
        )
        credential = None
        if self.sas_token:
            credential = self.sas_token
        elif self.account_key:
            credential = {
                'account_name': self.account_name,
                'account_key': self.account_key
            }
        file_client = ShareFileClient(
            account_url=account_url,
            credential=credential,
            share_name=self.container_name,
            file_path=src_file_path
        )

        future = tq.create_future(
            lambda: [
                (f.write(file_client.download_file().readall()), f.close())
                for f in [open(file_path, 'wb')]
            ][0][0]
        )
        future_handler = self._get_task_handler(src_file_path, counter, total, show_progress, "Download")
        async_task = AsyncTask(future, handler=future_handler, _ident="task_download_{}".format(src_file_path),
                               _parent_logger=module_logger)
        tq.add_task(async_task)
        return True

    def _file_share_upload(self, target, source):
        # file share does not automatically create directory
        dirs_to_create = []
        prev_dirpath = None
        dirpath = os.path.split(target)[0]
        account_url = "https://{account_name}.file.{endpoint}".format(
            account_name=self.account_name, endpoint=self.endpoint
        )
        credential = None
        if self.sas_token:
            credential = self.sas_token
        elif self.account_key:
            credential = {
                'account_name': self.account_name,
                'account_key': self.account_key
            }
        while dirpath != prev_dirpath and os.path.dirname(dirpath) != dirpath:
            file_service = ShareDirectoryClient(
                account_url=account_url,
                credential=credential,
                share_name=self.container_name,
                directory_path=dirpath
            )
            if not file_service.exists():
                dirs_to_create.append(dirpath)
                prev_dirpath = dirpath
                dirpath = os.path.split(dirpath)[0]
            else:
                break

        for dirpath in dirs_to_create[::-1]:
            file_service = ShareDirectoryClient(
                account_url=account_url,
                credential=credential,
                share_name=self.container_name,
                directory_path=dirpath
            )
            file_service.create_directory()

        def upload():
            file_account_url = "https://{account_name}.file.{endpoint}".format(
                account_name=self.account_name, endpoint=self.endpoint
            )
            service = ShareDirectoryClient(
                account_url=file_account_url,
                credential=credential,
                share_name=self.container_name,
                directory_path=os.path.split(target)[0]
            )
            with open(source, 'rb') as data:
                service.upload_file(
                    file_name=os.path.split(target)[1],
                    data=data
                )

        return upload

    @staticmethod
    def _verify_prefix(prefix):
        if not prefix:
            return
        prefix = prefix.lstrip("./\\")
        prefix_segments = re.split(r'[/\\]+', prefix)
        if len(prefix_segments) > 1:
            raise UserErrorException("Nested prefix '{}' for Azure File Share is currently not supported.")


class AzureBlobDatastore(AbstractAzureStorageDatastore):
    """Represents a datastore that saves connection information to Azure Blob storage.

    You should not work with this class directly. To create a datastore of this type, use
    the :meth:`azureml.core.datastore.Datastore.register_azure_blob_container` method of Datastore.

    Note: When using a datastore to access data, you must have permission to access that data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: azureml.core.workspace.Workspace
    :param name: The name of the datastore. It can only contain alphanumeric
        characters or - or _.
    :type name: str
    :param container_name: The blob container name.
    :type container_name: str
    :param account_name: The storage account name.
    :type account_name: str
    :param sas_token: The SAS token for accessing this container. Defaults to None.
    :type sas_token: str, optional
    :param account_key: The storage account key. Defaults to None.
    :type account_key: str, optional
    :param protocol: The protocol to use to connect to the storage account. If None, defaults to https.
    :type protocol: str, optional
    :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
    :type endpoint: str, optional
    :param request_session: the session object to use for http requests. Defaults to None.
    :type request_session: requests.Session, optional
    """

    def __init__(self, workspace, name, container_name, account_name, sas_token=None,
                 account_key=None, protocol=None, endpoint=None, request_session=None, workspace_msi_has_access=False,
                 subscription_id=None, resource_group=None, service_data_access_auth_identity=None):
        """Initialize a new Azure Blob Datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the datastore. It can only contain alphanumeric
            characters or - or _.
        :type name: str
        :param container_name: The blob container name.
        :type container_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param sas_token: The SAS token for accessing this container. Defaults to None.
        :type sas_token: str, optional
        :param account_key: The storage account key. Defaults to None.
        :type account_key: str, optional
        :param protocol: The protocol to use to connect to the storage account. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param request_session: The session object to use for http requests. Defaults to None.
        :type request_session: requests.Session, optional
        :param workspace_msi_has_access: Whether the Workspace MSI has access to this storage account.(Deprecated)
        :type workspace_msi_has_access: bool, optional
        :param subscription_id: The subscription ID of the storage account.
        :type subscription_id: str, optional
        :param resource_group: The resource group of the storage account.
        :type resource_group: str, optional
        :param service_data_access_auth_identity: Indicates which identity to use
            to authenticate service data access to customer's storage. Possible values
            include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
        :type service_data_access_auth_identity: str or
            ~_restclient.models.ServiceDataAccessAuthIdentity
        """
        super(AzureBlobDatastore, self).__init__(workspace, name, 'AzureBlob', container_name, account_name,
                                                 sas_token, account_key, protocol, endpoint)
        account_url = "https://{account_name}.blob.{endpoint}".format(
            account_name=self.account_name, endpoint=endpoint
        )
        credential = None
        if self.sas_token:
            credential = self.sas_token
        elif self.account_key:
            credential = {
                'account_name': self.account_name,
                'account_key': self.account_key
            }
        self.blob_service = BlobServiceClient(
            account_url=account_url,
            credential=credential
        )
        self.workspace_msi_has_access = workspace_msi_has_access
        self.service_data_access_auth_identity = service_data_access_auth_identity
        self.subscription_id = subscription_id
        self.resource_group = resource_group

    def download(self, target_path, prefix=None, overwrite=False, show_progress=True):
        """Download the data from the blob container to the local file system.

        :param target_path: The local directory to download the file to.
        :type target_path: str
        :param prefix: The path to the folder in the blob container to
            download. If set to None, will download everything in the blob. Defaults to None.
        :type prefix: str, optional
        :param overwrite: Indicates whether to overwrite existing files. Defaults to False.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show the progress of the download in the console.
            Defaults to True.
        :type show_progress: bool, optional
        :return: The number of files successfully downloaded.
        :rtype: int
        """
        def download(blob, dest):
            return lambda: [
                (f.write(self.blob_service.get_blob_client(
                    container=self.container_name,
                    blob=blob.name
                ).download_blob().readall()), f.close()) for f in [open(dest, 'wb')]
            ][0][0]

        module_logger.info("Called AzureBlobDatastore.download")
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        try:
            blobs = list(self.blob_service.get_container_client(
                container=self.container_name
            ).list_blobs(name_starts_with=prefix))
        except AzureHttpError as e:
            if e.status_code == 404 and self.credential_type == "None":
                raise ValueError("Download is not supported for datastores without credentials. To download your data "
                                 "please create an AzureML dataset and use download on the dataset.")
            raise e

        blobs = AzureBlobDatastore._filter_conflicting_blobs(blobs)
        total = len(blobs)
        if total == 0:
            return 0
        counter = _Counter()
        with TaskQueue(flush_timeout_seconds=float('inf'), _ident=__name__, _parent_logger=module_logger) as tq:
            for blob in blobs:
                file_path = os.path.join(target_path, os.path.normpath(blob.name))
                if not overwrite and os.path.exists(file_path):
                    module_logger.warning("Path already exists. Skipping download for {}".format(file_path))
                    total -= 1
                    continue

                dirname = os.path.dirname(file_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                future = tq.create_future(download(blob, file_path))
                future_handler = self._get_task_handler(blob.name, counter, total, show_progress, "Download")
                async_task = AsyncTask(future, handler=future_handler, _ident="task_download_{}".format(blob.name),
                                       _parent_logger=module_logger)
                tq.add_task(async_task)
        module_logger.info("Finished AzureBlobDatastore.download. Downloaded {} files.".format(counter.count()))
        return counter.count()

    @deprecated(
        '"Datastore.upload"',
        '"Dataset.File.upload_directory" to upload your files \
            from a local directory and create FileDataset in single method call')
    def upload(self, src_dir, target_path=None, overwrite=False, show_progress=True):
        """Upload the data from the local file system to blob container this data store points to.

        Note: For this method to work, the datastore object must have either account key or SAS
        token property populated. If datastore object doesn't contain any form of credential, this
        method will fail.
        Moreover, this method has been deprecated, may be removed in future. We suggest that you use
        the ``Dataset.File.upload_directory`` method to upload your files from local directory and
        create FileDataset in a single method call.

        :param src_dir: The local directory to upload.
        :type src_dir: str
        :param target_path: The location in blob container to upload to. If None, then upload to
           root. Defaults to None.
        :type target_path: str
        :param overwrite: Indicates whether to overwrite existing files. Defaults to False.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show progress of the upload in the console.
           Defaults to True.
        :type show_progress: bool, optional
        :return: The DataReference instance for the target path uploaded.
        :rtype: azureml.data.data_reference.DataReference
        """
        module_logger.info("Called AzureBlobDatastore.upload")
        self._ensure_credential("Upload")
        target_path = target_path or ""
        count = self._start_upload_task(
            self._get_upload_from_dir(src_dir, target_path),
            overwrite,
            lambda target_file_path: self.blob_service.get_blob_client(
                container=self.container_name,
                blob=target_file_path
            ).exists(),
            show_progress,
            lambda target, source: lambda: [(self.blob_service.get_blob_client(
                container=self.container_name,
                blob=target
            ).upload_blob(
                f, blob_type="BlockBlob", overwrite=overwrite
            ), f.close()) for f in [open(source, 'rb')]][0][0]
        )
        module_logger.info("Finished AzureBlobDatastore.upload with count={0}.".format(count))
        return DataReference(datastore=self, path_on_datastore=target_path)

    @deprecated(
        '"datastore.upload_files"',
        '"FileDatasetFactory.upload_directory" instead')
    def upload_files(self, files, relative_root=None, target_path=None, overwrite=False, show_progress=True):
        """Upload the data from the local file system to the blob container this datastore points to.

        Note: This method is deprecated and will no longer be supported. Recommend to use
        ``FileDatasetFactory.upload_directory`` instead.

        :param files: A list of absolute paths of files to upload.
        :type files: builtin.list[str]
        :param relative_root: The root used to determine the path
            of the files in the blob. For example, if we upload /path/to/file.txt, and we define
            base path to be /path, when file.txt is uploaded to the blob storage, it will have
            the path of /to/file.txt. If target_path is also given, then it will be used as
            the prefix for the derived path from above. The base path must be a common path of
            all of the files, otherwise an exception will be thrown. Defaults to None, which will find
            the common path.
        :type relative_root: str, optional
        :param target_path: The location in the blob container to upload the data to.
            Defaults to None, the root.
        :type target_path: str, optional
        :param overwrite: Indicates whether to overwrite existing files. Defaults to False.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show progress of the upload in the console.
            Defaults to True.
        :type show_progress: bool, optional
        :return: The DataReference instance for the target path uploaded.
        :rtype: azureml.data.data_reference.DataReference
        """
        module_logger.info("Called AzureBlobDatastore.upload_files")
        target_path = target_path or ""
        relative_root = relative_root or common_path(files)
        count = self._start_upload_task(
            self._get_upload_from_files(files, target_path, relative_root, False),
            overwrite,
            lambda target_file_path: self.blob_service.get_blob_client(
                container=self.container_name,
                blob=target_file_path
            ).exists(),
            show_progress,
            lambda target, source: lambda: [
                (self.blob_service.get_blob_client(
                    container=self.container_name,
                    blob=target
                ).upload_blob(
                    f, blob_type="BlockBlob", overwrite=overwrite
                ), f.close()) for f in [open(source, 'rb')]
            ][0][0]
        )
        module_logger.info("Finished AzureBlobDatastore.upload with count={0}.".format(count))
        return DataReference(datastore=self, path_on_datastore=target_path)

    def _as_dict(self, hide_secret=True):
        output = super(AzureBlobDatastore, self)._as_dict(hide_secret)
        if self.workspace_msi_has_access:
            output["workspace_msi_has_access"] = self.workspace_msi_has_access

        if self.service_data_access_auth_identity:
            output["service_data_access_auth_identity"] = self.service_data_access_auth_identity

        if self.subscription_id:
            output["subscription_id"] = self.subscription_id

        if self.resource_group:
            output["resource_group"] = self.resource_group

        return output

    @staticmethod
    def _filter_conflicting_blobs(blobs):
        def conflict_path(segment, throw):
            if throw:
                raise RuntimeError('Found non empty file {} that has the same name as a folder.'.format(segment))
            else:
                module_logger.info(
                    'Found empty file {} that has the same name as a folder, skipping it.'.format(segment))

        trie_dict = {}
        # blobs can be iterated only once in new azure storage blob package v12
        blobs, backup_blobs = tee(blobs)

        for blob in blobs:
            segments = list(accumulate(re.split('/|\\\\', blob.name), lambda acc, cur: '{}/{}'.format(acc, cur)))

            for segment in segments[:-1]:
                if segment in trie_dict:
                    conflict_path(segment, throw=trie_dict[segment] == FILE)
                trie_dict[segment] = FOLDER

            if segments[-1] in trie_dict:
                conflict_path(segments[-1], throw=blob.size != 0)
            else:
                trie_dict[segments[-1]] = EMPTY_FILE if blob.size == 0 else FILE

        return list(filter(lambda blob: trie_dict[blob.name] != FOLDER, backup_blobs))

    def _ensure_credential(self, action):
        if self.credential_type == "None":
            raise UserErrorException("{} cannot be performed since the Azure Storage Blob datastore ".format(action)
                                     + "does not have an access key (account key) or SAS token. "
                                     "Please make sure the datastore has an access key or SAS token.")


class _Counter(object):
    def __init__(self):
        self._count = 0

    def increment(self, by=1):
        self._count += by

    def count(self):
        return self._count
