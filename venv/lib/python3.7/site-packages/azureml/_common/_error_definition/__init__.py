# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._base_sdk_common import __version__ as VERSION
from .azureml_error import AzureMLError
from .user_error import UserError
from .system_error import SystemError
from .utils.error_decorator import error_decorator


__version__ = VERSION

__all__ = [
    'AzureMLError',
    'UserError',
    'SystemError',
    'error_decorator',
]
