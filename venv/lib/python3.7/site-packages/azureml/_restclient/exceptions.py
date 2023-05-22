# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import sys
import traceback

from .models.error_response import ErrorResponseException


class AzureMLRestClientException(Exception):
    """
    The base class for all AzureML REST-related exceptions.
    """

    def __init__(self, exception_message, inner_exception=None, **kwargs):
        Exception.__init__(self, exception_message, **kwargs)
        self._exception_message = exception_message
        self._inner_exception = inner_exception
        self._exc_info = sys.exc_info()

    @property
    def message(self):
        return self._exception_message

    @message.setter
    def message(self, value):
        self._exception_message = value

    @property
    def inner_exception(self):
        return self._inner_exception

    @inner_exception.setter
    def inner_exception(self, value):
        self._inner_exception = value

    def print_stacktrace(self):
        traceback.print_exception(*self._exc_info)


class ServiceException(AzureMLRestClientException, ErrorResponseException):
    """
    An exception related to failed service calls. The exception contains the status code and
    any details if available related to the exception.
    """

    def __init__(self, error_response_exception, **kwargs):
        # Extending ServiceException to preserve all of the attributes of ErrorResponseException
        for key, value in error_response_exception.__dict__.items():
            setattr(self, key, value)
        # Errors created in the networking layer won't populate error_response_exception.error.error
        has_error = hasattr(error_response_exception, "error") and hasattr(error_response_exception.error, "error")
        has_error = has_error and hasattr(error_response_exception.error.error, "details")
        self.details = error_response_exception.error.error.details if has_error else []
        self.status_code = error_response_exception.response.status_code

        AzureMLRestClientException.__init__(self, error_response_exception.message, error_response_exception)

    def __repr__(self):
        details_message = ""
        for error_detail in self.details:
            details_message += "\t\t{}\n".format(error_detail.message)

        try:
            formatted_headers = json.dumps(dict(self._inner_exception.response.headers), indent=4)
            formatted_headers = "\t".join(formatted_headers.splitlines(keepends=True))  # Extra indent on each line.
        except Exception:  # Maybe e.response doesn't exist or isn't the kind of object we expect.
            formatted_headers = ""

        return "ServiceException:\n\tCode: {}\n\tMessage: {}\n\tDetails:\n{}\n\tHeaders: {}\n\tInnerException: {}" \
            .format(self.status_code, self.message, details_message, formatted_headers, self.inner_exception.error)

    def __str__(self):
        return self.__repr__()
