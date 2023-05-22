# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import hashlib
import base64
import os
import logging
import time
import re

from requests import Session
from requests.exceptions import RequestException
from urllib3.exceptions import HTTPError

from azureml._common.exceptions import AzureMLException
from azureml._vendor.azure_storage.blob import BlobServiceClient
from azureml._file_utils.upload import get_block_blob_service_credentials
from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import BadDataDownloaded, DownloadFailed, BlobNotFound
from azure.common import AzureException
from azure.common import AzureMissingResourceHttpError

module_logger = logging.getLogger(__name__)


def _validate_content_match(server_md5, computed_md5):
    _ERROR_MD5_MISMATCH = \
        'MD5 mismatch. Expected value is \'{0}\', computed value is \'{1}\'.'
    if server_md5 != computed_md5:
        raise AzureMLException(_ERROR_MD5_MISMATCH.format(server_md5, computed_md5))


def normalize_path_and_join(path, name):
    """
    Normalizes user provided paths by expanding the user paths such as ~ or ~user,
    converting it into an absolute path and joins the path with the provided file or directory name.
    :param path: Path to normalize.
    :type path: str
    :param name: Name of the file(with extension) or directory name.
    :type name: str
    :return: A normalized absolute path, including the file name or directory name.
    :rtype: str
    """
    # Expand the user path if the path starts with '~' or '~user'.
    normalized_path = normalize_path(path)
    if os.path.basename(normalized_path) != name:
        normalized_path = os.path.join(normalized_path, name)
    return normalized_path


def normalize_path(path):
    """
    Normalizes user provided paths by expanding the user paths such as ~ or ~user
    and converting it into an absolute path.
    :param path: Path to normalize.
    :type path: str
    :return: A normalized, absolute path.
    :rtype: str
    """
    return os.path.abspath(os.path.expanduser(path))


def directory_exists(path, directory_name):
    """
    Normalizes the path and checks if the directory exists.
    :param path: Path to check for the directory.
    :type path: str
    :param directory_name: Name of the directory to check.
    :type directory_name: str
    :return: True or False based on whether the directory exists.
    :rtype: bool
    """
    normalized_path = normalize_path_and_join(path, directory_name)
    return os.path.isdir(normalized_path)


def check_and_create_dir(path, directory_name=None):
    """
    Normalizes the provided path and creates the directory if it doesn't exist.
    :param path: Path to create the directory in.
    :type path: str
    :param directory_name: Name of the directory to create.
    :type directory_name: str
    """
    if directory_name is None:
        # directory_name hasn't been provided, so use the basename of the provided
        # path as the directory name.
        directory_name = os.path.basename(path)

    if not directory_exists(path, directory_name):
        # Directory doesn't exist, so create the directory
        normalized_path = normalize_path_and_join(path, directory_name)
        os.mkdir(normalized_path)


def makedirs_for_file_path(file_path):
    """
    :param file_path: relative or absolute path to a file
    """
    parent_path = os.path.join(file_path, os.path.pardir)
    parent_path = os.path.normpath(parent_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)
    return True


def get_root_path():
    """
    Gets the root directory for the drive.
    NOTE: On Windows, it returns 'C:\' or the path to the root dir for the drive.
          On Linux, it returns '/'.
    :return: Path to the root directory for the drive.
    :rtype: str
    """
    return os.path.realpath(os.sep)


def traverse_up_path_and_find_file(path, file_name, directory_name=None, num_levels=None):
    """
    Traverses up the provided path until we find the file, reach a directory
    that the user does not have permissions to, or if we reach num_levels (if set by the user).
    NOTE: num_levels=2 would mean that we search the current directory and two levels above (inclusive).
    :param path: Path to traverse up from.
    :type path: str
    :param file_name: The name of the file to look for, including the file extension.
    :type file_name: str
    :param directory_name: (optional)The name of the directory that the file should be in. ie) /aml_config/config.json
    :type directory_name: str
    :param num_levels: Number of levels to traverse up the path for (inclusive).
    :type num_levels: int
    :return: Path to the file that we found, or an empty string if we couldn't find the file.
    :rtype: str
    """
    current_path = normalize_path(path)
    if directory_name is not None:
        file_name = os.path.join(directory_name, file_name)

    current_level = 0
    root_path = get_root_path()
    while True:
        path_to_check = os.path.join(current_path, file_name)
        if os.path.isfile(path_to_check):
            return path_to_check

        if current_path == root_path or (num_levels is not None and num_levels == current_level):
            break
        current_path = os.path.realpath(os.path.join(current_path, os.path.pardir))
        current_level = current_level + 1

    return ''


def normalize_file_ext(file_name, extension):
    """
    Normalizes the file extension by appending the provided extension to the file_name.
    If file_name contains the file extension, we make sure that it matches the extension provided (case-sensitive).

    :param file_name: The name of the file to normalize (may or may not contain the file extension).
    :type file_name: str
    :param extension: File extension to use for the file, with or without the leading period (i.e. '.json' or 'json').
    :type extension: str
    :return: File name and the extension for the file.
    :rtype: str
    """
    extension = extension if extension[0] == '.' else '.' + extension
    root, ext = os.path.splitext(file_name)
    if not ext:
        # Case when file_name doesn't contain the file extension and ext is an empty string.
        return file_name + extension
    return root + extension


def download_file(source_uri, path=None, max_retries=5, stream=True, protocol="https", session=None,
                  _validate_check_sum=False, max_concurrency=8, fail_on_not_found=False):
    """
    Downloads the file from source_uri. Saves the content to the path if set.

    :param source_uri: The name of the file to normalize (may or may not contain the file extension).
    :type source_uri: str
    :param path: if set the content of the file will be written to path
    :type path: str
    :param max_retries: the number of retries
    :type max_retries: int
    :param stream: Whether to incrementally download the file
    :type stream: bool
    :param protocol: The http protocol for the get request, defaults to https
    :type protocol: str
    :param session: the shared session from the caller
    :type session: requests.session
    :param _validate_check_sum: to validate the content from HTTP response
    :type _validate_check_sum: bool
    :param max_concurrency: default value: 1
    :type max_concurrency: int
    :param fail_on_not_found: optional flag to raise exception if BlobNotFound instead of retrying
    :type fail_on_not_found: bool
    :return: None
    :rtype: NoneType
    """
    module_logger.debug("downloading file to {path}, with max_retries: {max_retries}, "
                        "stream: {stream}, and protocol: {protocol}".format(path=path,
                                                                            max_retries=max_retries,
                                                                            stream=stream,
                                                                            protocol=protocol))
    if path is None:
        module_logger.debug('Output file path is {}, the file was not downloaded.'.format(path))
        return

    # download using BlobClient
    if is_source_uri_matches_storage_blob(source_uri):
        sas_token, account_name, endpoint_suffix, container_name, blob_name = get_block_blob_service_credentials(
            source_uri)
        account_url = "https://{account_name}.blob.{endpoint}".format(
            account_name=account_name, endpoint=endpoint_suffix
        )
        blob_service = BlobServiceClient(
            account_url=account_url,
            credential=sas_token
        )
        makedirs_for_file_path(path)

        def exec_func():
            metadata_dict = blob_service.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            with open(path, "wb") as my_blob:
                download_stream = metadata_dict.download_blob(
                    max_concurrency=max_concurrency, validate_content=_validate_check_sum
                )
                my_blob.write(download_stream.readall())
                content_length = download_stream.size
            file_size = os.stat(path).st_size
            module_logger.debug("Downloaded file {} with size {}.".format(path, file_size))
            if(content_length != file_size):
                azureml_error = AzureMLError.create(
                    BadDataDownloaded, file_size=file_size, content_length=content_length
                )
                raise AzureMLException._with_error(azureml_error)

        fail_on_exceptions = ()
        if fail_on_not_found:
            fail_on_exceptions = (AzureMissingResourceHttpError)
        return _retry(exec_func, max_retries=max_retries, fail_on_exceptions=fail_on_exceptions)

    # download using requests.Session
    def _handle_response(response):
        makedirs_for_file_path(path)
        md5_hash = hashlib.md5()
        with open(path, 'wb') as write_to_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    md5_hash.update(chunk)
                    write_to_file.write(chunk)
            if _validate_check_sum:
                _validate_content(md5_hash, response)

    _request_file_with_retry(source_uri, _handle_response, max_retries, stream, session)


def download_file_stream(source_uri, encoding="utf-8", download_to_bytes=False, max_retries=5, stream=True,
                         protocol="https", session=None, _validate_check_sum=False, max_concurrency=8):
    """
    Downloads the file from source_uri. Returns the content of the response and decoded to user specified encoding"

    :param source_uri: The name of the file to normalize (may or may not contain the file extension).
    :type source_uri: str
    :param encoding: encoding of the http body
    :type encoding: str
    :param download_to_bytes: ignore encoding and directly download to bytes
    :type download_to_bytes: bool
    :param max_retries: the number of retries
    :type max_retries: int
    :param stream: Whether to incrementally download the file
    :type stream: bool
    :param protocol: The http protocol for the get request, defaults to https
    :type protocol: str
    :param session: the shared session from the caller
    :type session: requests.session
    :return: the response content
    :rtype: str | bytes
    """

    # download using BlobClient
    if is_source_uri_matches_storage_blob(source_uri):
        sas_token, account_name, endpoint_suffix, container_name, blob_name = get_block_blob_service_credentials(
            source_uri)
        account_url = "https://{account_name}.blob.{endpoint}".format(
            account_name=account_name, endpoint=endpoint_suffix
        )
        blob_service = BlobServiceClient(
            account_url=account_url,
            credential=sas_token
        )

        def exec_func():
            if download_to_bytes:
                blob = blob_service.get_blob_client(
                    container=container_name,
                    blob=blob_name
                ).download_blob(
                    max_concurrency=max_concurrency,
                    validate_content=_validate_check_sum
                )
            else:
                blob = blob_service.get_blob_client(
                    container=container_name,
                    blob=blob_name
                ).download_blob(
                    max_concurrency=max_concurrency,
                    validate_content=_validate_check_sum,
                    encoding=encoding
                )
            return blob.readall()

        return _retry(exec_func, max_retries=max_retries)

    # download using requests.Session
    def _handle_response(response):
        bytes_str = bytes()
        md5_hash = hashlib.md5()
        if response.status_code != 200:
            response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                md5_hash.update(chunk)
                bytes_str += chunk
        if _validate_check_sum:
            _validate_content(md5_hash, response)
        return bytes_str if download_to_bytes else bytes_str.decode(encoding)

    return _request_file_with_retry(source_uri, _handle_response, max_retries, stream, session)


def _validate_content(md5_hash, response):
    """
    Validate the content of response with md5_hash

    :param md5_hash:
    :type md5_hash: _Hash
    :param response: the response object
    :type response: requests.Response
    :return: None
    :rtype: None
    """
    if 'content-md5' in response.headers:
        _validate_content_match(response.headers['content-md5'],
                                base64.b64encode(md5_hash.digest()).decode('utf-8'))
    else:
        module_logger.debug(
            "validate_check_sum flag is set to true but content-md5 not found on respose header")


def get_directory_size(path, size_limit=None, include_function=None, exclude_function=None):
    """
    Get the size of the directory. If size_limit is specified, stop after reaching this value.

    :type path: str
    :type include_function: Callable
    :type exclude_function: Callable

    :rtype: int
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            full_path = os.path.normpath(os.path.join(dirpath, name))

            if ((not exclude_function and not include_function)
                    or (exclude_function and not exclude_function(full_path))
                    or (include_function and include_function(full_path))):
                total_size += os.path.getsize(full_path)
                if size_limit and total_size > size_limit:
                    return total_size

    return total_size


def get_path_size(file_or_folder_path, size_limit, exclude_function=None):
    """
    Calculate the size of the file or folder
    :param file_or_folder_path:
    :type file_or_folder_path: str
    :rtype: int the size of the file or folder
    """
    if os.path.isfile(file_or_folder_path):
        size = os.path.getsize(file_or_folder_path)
    else:
        size = get_directory_size(
            file_or_folder_path, size_limit=size_limit, exclude_function=exclude_function)
    return size


def is_source_uri_matches_storage_blob(source_uri):
    """
    Regex matches the source_uri with azure storage blob url
    :param source_uri: The name of the file to normalize (may or may not contain the file extension).
    :type source_uri: str
    :return: true if regex matches successfully
    :rtype: bool
    """
    pattern = '^{}(.*){}(.*){}(.*){}(.*)'.format(re.escape("https://"),
                                                 re.escape(".blob.core.windows.net/"),
                                                 re.escape("/"),
                                                 re.escape("?"))
    return re.match(pattern, source_uri) is not None


def _request_file_with_retry(source_uri, handle_response, max_retries, stream, session):
    """
    Downloads the file from source_uri using requests session, and then call handle_response function on response.
    Returns the result from handle_response

    :param source_uri: The name of the file to normalize (may or may not contain the file extension).
    :type source_uri: str
    :param handle_response: a function that handles response
    :type handle_response: func
    :param max_retries: the number of retries
    :type max_retries: int
    :param stream: Whether to incrementally download the file
    :type stream: bool
    :param session: the requests session
    :type session: requests.session
    :return: Any result from handle_response
    :rtype: AnyType
    """
    is_new_session = session is None
    if is_new_session:
        module_logger.warning('requests session is not set.')
        session = Session()

    def exec_func():
        response = session.get(source_uri, stream=stream)
        if response.status_code != 200:
            response.raise_for_status()
        return handle_response(response)

    def clean_up_func():
        if is_new_session:
            session.close()

    exceptions = (RequestException, HTTPError, AzureException)
    return _retry(exec_func, clean_up_func, max_retries=max_retries, retry_on_exceptions=exceptions)


def _retry(exec_func, clean_up_func=(lambda: None), max_retries=5,
           retry_on_exceptions=(Exception), fail_on_exceptions=()):
    """
    A helper function for retry

    :param exec_func: the execution function that runs inside retry mechanism
    :type exec_func: func
    :param clean_up_func: a clean up function that runs inside final statement
    :type clean_up_func: func
    :param max_retries: the number of retries
    :type max_retries: int
    :param retry_on_exceptions: the exceptions to handle with retry in execution function
    :type stream: Tuple[Type[Exception]]
    :param fail_on_exceptions: the exceptions to wrap and raise with no retry
    :type stream: Tuple[Type[Exception]]
    :return: results from the return of execution func
    :rtype: AnyType
    """
    wait_time = 2
    retries = 0
    while retries < max_retries:
        try:
            return exec_func()
        except fail_on_exceptions as ex:
            # if condition to fail on different exceptions differently
            if isinstance(ex, AzureMissingResourceHttpError):
                # We don't want to retry when there is no resource to start with
                exc = AzureMLException._with_error(
                    AzureMLError.create(BlobNotFound)
                )
                # 'from exc' is added to suppress chained exception coming from AzureMissingResourceHttpError
                # which is has the same exception message we are raising
                raise exc from exc
        except retry_on_exceptions as request_exception:
            retries += 1
            module_logger.debug('retry has happened in the {} times'.format(retries))
            if retries < max_retries:
                module_logger.debug(
                    'RequestException or HTTPError raised in download_file with message: {}'.format(request_exception))
                time.sleep(wait_time)
                wait_time = wait_time ** 2
                continue
            else:
                module_logger.error('Failed to download file with error: {}'.format(request_exception))
                azureml_error = AzureMLError.create(
                    DownloadFailed, error=request_exception
                )
                raise AzureMLException._with_error(azureml_error)
        finally:
            clean_up_func()
