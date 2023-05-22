# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from abc import ABC

from azureml._common._error_definition.error_definition import ErrorDefinition
from azureml._common._error_definition.error_strings import AzureMLErrorStrings


class SystemError(ErrorDefinition, ABC):
    """Standard list of errors that can be caused due to invalid assumptions in the internal system."""
    pass


class ClientError(SystemError):
    """Errors caused internally by SDK.
    Usually an indication of a bug due to mis-configuration or an invalid assumption by the system."""
    @property
    def message_format(self) -> str:
        return AzureMLErrorStrings.SystemErrorStrings.CLIENT_ERROR
