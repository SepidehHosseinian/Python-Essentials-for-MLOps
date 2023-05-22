# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from azureml._common._error_definition.error_definition import ErrorDefinition


def error_decorator(use_parent_error_code=False, is_transient=False, details_uri=""):
    """
    Convenience decorator to be used on implementers of ErrorDefinition, to override some of the properties on
    the base class.

    :param use_parent_error_code: If True, will use the parent class's error code
    :param is_transient: Determines whether this error is transient and retry-able
    :param details_uri: Pointer to a web  URL indicating more details about the error
    :return: Wrapper for an ErrorDefinition class with overridden properties
    """
    def wrapper(cls):
        if not issubclass(cls, ErrorDefinition):
            raise ValueError("'error_decorator' is only intended to be used to with AzureML "
                             "Error Definitions.")
        if use_parent_error_code:
            setattr(cls, 'use_parent_error_code', property(lambda _: use_parent_error_code))
        if is_transient:
            setattr(cls, 'is_transient', property(lambda _: is_transient))
        if details_uri:
            setattr(cls, 'details_uri', property(lambda _: details_uri))
        return cls

    return wrapper
