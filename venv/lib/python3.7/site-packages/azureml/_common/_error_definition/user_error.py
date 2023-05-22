# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from abc import ABC

from azureml._common._error_definition.error_definition import ErrorDefinition
from azureml._common._error_definition.error_strings import AzureMLErrorStrings


class UserError(ErrorDefinition, ABC):
    """
    Standard list of errors that can be caused due to invalid user inputs.

    └── UserError                           Errors caused due to invalid inputs from the user
        |
        ├── BadArgument
        │   ├── ArgumentInvalid             Argument is of invalid type
        │   ├── ArgumentBlankOrEmpty        Argument provided was either null or empty
        │   ├── ArgumentOutOfRange          Argument is not within the permitted range
        │   ├── MalformedArgument           Argument format is not as expected
        │   ├── DuplicateArgument           Redundant argument is provided, often with different values
        │   └── ArgumentMismatch            An invalid combination of arguments are provided
        |
        ├── BadData                         User provided data is erroneous
        │   ├── EmptyData                   Entire input data provided by the user resulted in an empty dataset
        │   ├── MissingData                 Input data resulted in a non-empty dataset, but some key parts are missing
        │   ├── InvalidDimension            Input dataset has bad dimensions
        │   └── InvalidData                 Input dataset has bad content
        |
        ├── NotFound                        User provided resources were not found
        │   ├── WorkspaceNotFound
        │   ├── ExperimentNotFound
        │   ├── ComputeNotFound
        │   ├── KeyVaultNotFound
        │   └── StorageAccountNotFound
        |
        ├── Disabled                        User owned resource is disabled, so can't proceed
        |
        ├── Immutable                       User owned resource is immutable, so can't proceed
        |
        ├── Conflict                        User owned resource is in a conflicting state, so can't proceed
        |
        ├── NotReady                        User owned resource is not ready, so can't proceed
        |
        ├── Deprecated                      Requested feature or operation is deprecated
        |
        ├── NotSupported                    Requested feature or operation is not supported
        |
        ├── ConnectionFailure               Connection to user owned resource failed
        |
        ├── Auth                            Any Authorization or Authentication errors when accessing user resources
        │   ├── Authentication
        │   └── Authorization
        |
        ├── QuotaExceeded
        │   └── TooManyRequests
        |
        └── ResourceExhausted               Insufficient resources to complete the operation
            ├── Memory
            └── Timeout
    """
    pass


class BadArgument(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.BAD_ARGUMENT


class ArgumentInvalid(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ARGUMENT_INVALID


class ArgumentBlankOrEmpty(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ARGUMENT_BLANK_OR_EMPTY


class ArgumentOutOfRange(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ARGUMENT_OUT_OF_RANGE


class MalformedArgument(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.MALFORMED_ARGUMENT


class DuplicateArgument(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.DUPLICATE_ARGUMENT


class ArgumentMismatch(BadArgument):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.ARGUMENT_MISMATCH


class BadData(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.BAD_DATA


class EmptyData(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.EMPTY_DATA


class MissingData(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.MISSING_DATA


class InvalidDimension(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_DIMENSION


class InvalidData(BadData):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.INVALID_DATA


class NotFound(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.NOT_FOUND


class WorkspaceNotFound(NotFound):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.WORKSPACE_NOT_FOUND


class ExperimentNotFound(NotFound):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.EXPERIMENT_NOT_FOUND


class ComputeNotFound(NotFound):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.COMPUTE_NOT_FOUND


class KeyVaultNotFound(NotFound):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.KEY_VAULT_NOT_FOUND


class StorageAccountNotFound(NotFound):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.STORAGE_ACCOUNT_NOT_FOUND


class Disabled(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.DISABLED


class Immutable(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.IMMUTABLE


class Conflict(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.CONFLICT


class NotReady(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.NOT_READY


class Deprecated(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.DEPRECATED


class NotSupported(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.NOT_SUPPORTED


class ArgsNotSupportedForScenario(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.NOT_SUPPORTED_FOR_SCENARIO


class ConnectionFailure(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.CONNECTION_FAILURE


class Auth(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.AUTH


class Authentication(Auth):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.AUTHENTICATION


class Authorization(Auth):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.AUTHORIZATION


class QuotaExceeded(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.QUOTA_EXCEEDED


class TooManyRequests(QuotaExceeded):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.TOO_MANY_REQUESTS


class ResourceExhausted(UserError):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.RESOURCE_EXHAUSTED


class Memory(ResourceExhausted):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.MEMORY


class Timeout(ResourceExhausted):
    @property
    def message_format(self):
        return AzureMLErrorStrings.UserErrorStrings.TIMEOUT
