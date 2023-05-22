# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._base_sdk_common import __version__ as VERSION

from .file_utils import download_file, download_file_stream, get_path_size
from .upload import upload_blob_from_stream, get_block_blob_service_credentials

# TODO add rest of the methods in file_utils

__version__ = VERSION
__all__ = ["download_file", "download_file_stream", "upload_blob_from_stream", "get_block_blob_service_credentials",
           "get_path_size"]
