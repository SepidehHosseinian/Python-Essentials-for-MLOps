# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from azureml._common._error_definition import error_decorator
from azureml._common._error_definition.error_strings import AzureMLErrorStrings

from azureml._common._error_definition.user_error import (
    InvalidData,
    Timeout,
    BadArgument,
    BadData,
    Authorization,
    ConnectionFailure,
    ArgumentInvalid,
    Authentication,
    NotFound
)


@error_decorator(use_parent_error_code=True)
class BlobNotFound(NotFound):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.BLOB_NOT_FOUND


@error_decorator(use_parent_error_code=True)
class ArgumentSizeOutOfRangeType(ArgumentInvalid):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ARGUMENT_SIZE_OUT_OF_RANGE_TYPE


@error_decorator(use_parent_error_code=True)
class DownloadFailed(ConnectionFailure):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.DOWNLOAD_FAILED


@error_decorator(use_parent_error_code=True)
class InvalidColumnLength(InvalidData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_COLUMN_LENGTH


@error_decorator(use_parent_error_code=True)
class FlushTaskTimeout(Timeout):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.TIMEOUT_FLUSH_TASKS


@error_decorator(use_parent_error_code=True)
class BadDataDownloaded(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.BAD_DATA_DOWNLOAD


@error_decorator(use_parent_error_code=True)
class BadDataUpload(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.BAD_DATA_UPLOAD


@error_decorator(use_parent_error_code=True)
class AuthorizationStorageAccount(Authorization):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.AUTHORIZATION_BLOB_STORAGE


@error_decorator(use_parent_error_code=True)
class InvalidArgumentType(ArgumentInvalid):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ARGUMENT_INVALID_TYPE


@error_decorator(use_parent_error_code=True)
class InvalidColumnData(InvalidData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_COLUMN_DATA


@error_decorator(use_parent_error_code=True)
class InvalidOutputStream(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_OUTPUT_STREAM


@error_decorator(use_parent_error_code=True)
class InvalidUri(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_URI


@error_decorator(use_parent_error_code=True)
class InvalidStatus(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_STATUS


@error_decorator(use_parent_error_code=True)
class MethodAlreadyRegistered(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.METHOD_ALREADY_REGISTERED


@error_decorator(use_parent_error_code=True)
class MethodNotRegistered(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.METHOD_NOT_REGISTERED


@error_decorator(use_parent_error_code=True)
class StartChildrenFailed(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.START_CHILDREN_FAILED


@error_decorator(use_parent_error_code=True)
class CreateChildrenFailed(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.CREATE_CHILDREN_FAILED


@error_decorator(use_parent_error_code=True)
class TwoInvalidArgument(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.TWO_INVALID_ARGUMENT


@error_decorator(use_parent_error_code=True)
class TwoInvalidParameter(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.TWO_INVALID_PARAMETER


@error_decorator(use_parent_error_code=True)
class UnsupportedReturnType(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.UNSUPPORTED_RETURN_TYPE


@error_decorator(use_parent_error_code=True)
class FileAlreadyExists(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.FILE_ALREADY_EXISTS


@error_decorator(use_parent_error_code=True)
class FailedIdWithinSeconds(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.FAILED_ID_WITHIN_SECONDS


@error_decorator(use_parent_error_code=True)
class OnlySupportedServiceSideFiltering(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ONLY_SUPPORTED_SERVICE_SIDE_FILTERING


@error_decorator(use_parent_error_code=True)
class MetricsNumberExceeds(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.METRICS_NUMBER_EXCEEDS


@error_decorator(use_parent_error_code=True)
class CredentialsExpireInactivity(Authentication):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.CREDENTIAL_EXPIRED_INACTIVITY


@error_decorator(use_parent_error_code=True)
class AccountConfigurationChanged(Authentication):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ACCOUNT_CONFIGURATION_CHANGED


@error_decorator(use_parent_error_code=True)
class CredentialExpiredPasswordChanged(Authentication):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.CREDENTIAL_EXPIRED_PASSWORD_CHANGE


@error_decorator(use_parent_error_code=True)
class CertificateVerificationFailure(Authentication):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.SSL_ERROR


@error_decorator(use_parent_error_code=True)
class NetworkConnectionFailed(ConnectionFailure):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.NETWORK_CONNECTION_FAILURE
