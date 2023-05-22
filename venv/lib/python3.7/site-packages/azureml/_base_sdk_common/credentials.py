# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" credentials.py, methods for interacting with the AzureML credential service."""

from __future__ import print_function

import requests


def set_credentials(experiment, key, value):
    workspace_uri_path = experiment.workspace.service_context._get_workspace_scope()
    address = experiment.workspace.service_context._get_run_history_url()
    address += "/credential/v1.0" + workspace_uri_path + "/secrets"

    headers = experiment.workspace._auth_object.get_authentication_header()

    body = {"Name": key, "Value": value}
    response = requests.put(address, json=body, headers=headers)
    response.raise_for_status()


def get_credentials(experiment, credential_name):

    import six.moves.urllib as urllib
    workspace_uri_path = experiment.workspace.service_context._get_workspace_scope()
    address = experiment.workspace.service_context._get_run_history_url()
    encoded_credential = urllib.parse.quote_plus(credential_name)
    address += "/credential/v1.0" + workspace_uri_path + "/secrets/" + encoded_credential

    headers = experiment.workspace._auth_object.get_authentication_header()
    response = requests.get(address, headers=headers)
    response.raise_for_status()
    return response.json()["value"]


def remove_credentials(experiment, key):
    # TODO Implement this once the service supports deletion.
    pass
