# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Internal use only."""

from functools import wraps

from azureml.exceptions import UserErrorException
from azureml._restclient.models.error_response import ErrorResponseException


def _handle_error_response_exception(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ErrorResponseException as e:
            if e.response.status_code < 500 and "(UserError)" in str(e):
                raise UserErrorException(str(e))
            raise e

    return decorated
