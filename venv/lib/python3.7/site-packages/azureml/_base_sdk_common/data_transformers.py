# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from collections import OrderedDict


def transform_list_to_tabular_data(results):
    """ A utility function that transforms captioned (with caption being the
        first list entry) list of items into tabular data expected by the
        Azure CLI table output
    """
    if len(results) <= 1:
        return None

    transformed = []
    captions = results[0]
    for i in range(1, len(results)):
        entry = OrderedDict()
        for j in range(len(captions)):
            entry[captions[j]] = results[i][j]
        transformed.append(entry)

    return transformed


def transform_project_data(results, header):
    """ A utility function for transforming single list items to
     key value pairs and sending them to transform_list_to_tabular_data"""

    if len(results) <= 1:
        return None

    transformed = []

    for item in results:
        entry = OrderedDict([(header, item), ('Value', '')])
        transformed.append(entry)

    return transformed


def compute_context_transformer(values):
    """Calls transform_project_data with corresponding header"""
    return transform_project_data(values, "Compute Target")


def runconfiguration_transformer(values):
    """Calls transform_project_data with corresponding header"""
    return transform_project_data(values, "Run configuration")
