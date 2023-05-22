# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
from .utils import _get_codes


class ErrorHierarchy:
    """
    Object to represent the error hierarchy json for an error.
    """

    def __init__(self, error_response):
        if isinstance(error_response, str):
            self.error = json.loads(error_response)
        elif isinstance(error_response, dict):
            self.error = error_response
        else:
            raise ValueError("Input error_response should be a dict or json string.")

        self.error_list = _get_codes(self.error)
        self.error_list.reverse()

    def __repr__(self):
        """Return the error hierarchy in increasing granularity.

        :return: String representation of the ErrorHierarchy object
        :rtype: str
        """
        return '.'.join(self.error_list) if self.error_list is not None else None

    def __str__(self):
        """Return the string representation of the ErrorHierarchy."""
        return self.__repr__()

    def get_root(self):
        return self.error_list[0] if self.error_list is not None else None

    def get_leaf(self):
        return self.error_list[-1] if self.error_list is not None else None
