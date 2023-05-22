# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
from azureml._common._error_response import _error_response_constants


def _get_codes(error_response_json):
    """Get the list of error codes from an error response json"""
    if isinstance(error_response_json, str):
        error_response_json = json.loads(error_response_json)
    error_response_json = error_response_json.get("error")
    if error_response_json is None:
        return []
    code = error_response_json.get('code')
    if code is None:
        raise ValueError("Error response does not contain an error code.")
    codes = [code]
    inner_error = error_response_json.get(
        'inner_error', error_response_json.get('innerError', None))
    while inner_error is not None:
        code = inner_error.get('code')
        if code is None:
            break
        codes.append(code)
        inner_error = inner_error.get(
            'inner_error', inner_error.get('innerError', None))
    return codes[::-1]


def code_in_error_response(error_response, error_code):
    """Given an error response, returns whether a code exists in it."""
    if error_response is None or error_response == "null":
        return False
    try:
        error_response_json = json.loads(error_response)
        codes = _get_codes(error_response_json)
        return error_code in codes
    except Exception:
        return False


def _code_in_hierarchy(leaf_code, target_code):
    """
    Given a leaf node, check if a target code is in the hierarchy. This is limited to hierarchies this
    version of the SDK is aware of.
    """
    for _, hierarchy in vars(_error_response_constants.ErrorHierarchy).items():
        if isinstance(hierarchy, list) and leaf_code in hierarchy:
            return target_code in hierarchy
    return False


def is_error_code(exception, error_code_hierarchy):
    """
    Determine whether an error code is in the exceptions error code hierarchy.
    :param exception: exception to check for error code hierarchy
    :param error_code_hierarchy: The desired code hierarchy as found in
        azureml._common_error_response._error_response_constants.ErrorHierarchy
    :return: bool
    """
    # Import at runtime due to circular dependency in azureml._common.exceptions and
    # azureml._common._error_response.error_hierarchy
    from azureml._common.exceptions import AzureMLException

    if isinstance(exception, AzureMLException):
        error_response = exception._serialize_json()
        for error_code in error_code_hierarchy:
            if not code_in_error_response(error_response, error_code):
                return False
        return True

    if hasattr(exception, "_error_code"):
        return exception._error_code == error_code_hierarchy

    error_response = None
    if hasattr(exception, "error"):
        try:
            error_response = json.loads(str(exception.error))
        except Exception:
            pass

    if error_response is None:
        # Note that this is a hack since exception received from service side doesnt have "error" or "_error_code"
        # Till we fix that this is workaround.
        try:
            error_response = json.loads(exception.response.content)
        except Exception:
            pass

        try:
            for error_code in error_code_hierarchy:
                if not code_in_error_response(json.dumps(error_response), error_code):
                    return False
            return True
        except ValueError:
            pass

    return False
