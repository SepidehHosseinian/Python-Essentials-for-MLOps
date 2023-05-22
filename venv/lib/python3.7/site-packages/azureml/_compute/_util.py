# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import isodate
import requests
from math import floor
from pkg_resources import resource_string
from azureml.exceptions import ComputeTargetException
from azureml.exceptions import UserErrorException
from azureml._restclient.clientbase import ClientBase
from azureml._base_sdk_common.common import to_camel_case


amlcompute_payload_template = json.loads(resource_string(__name__, 'data/amlcompute_cluster_template.json')
                                         .decode('ascii'))
computeinstance_payload_template = json.loads(resource_string(__name__, 'data/computeinstance_payload_template.json')
                                              .decode('ascii'))
aks_payload_template = json.loads(resource_string(__name__, 'data/aks_cluster_template.json').decode('ascii'))
kubernetes_compute_template = json.loads(resource_string(__name__, 'data/kubernetes_compute_template.json')
                                         .decode('ascii'))
dsvm_payload_template = json.loads(resource_string(__name__, 'data/dsvm_cluster_template.json').decode('ascii'))
hdinsight_payload_template = json.loads(resource_string(__name__, 'data/hdinsight_cluster_template.json')
                                        .decode('ascii'))
datafactory_payload_template = json.loads(resource_string(__name__, 'data/datafactory_payload_template.json')
                                          .decode('ascii'))
databricks_compute_template = json.loads(resource_string(__name__, 'data/databricks_compute_template.json')
                                         .decode('ascii'))
adla_payload_template = json.loads(resource_string(__name__, 'data/adla_payload_template.json').decode('ascii'))
remote_payload_template = json.loads(resource_string(__name__, 'data/remote_compute_template.json').decode('ascii'))
batch_compute_template = json.loads(resource_string(__name__, 'data/batch_compute_template.json').decode('ascii'))
kusto_compute_template = json.loads(resource_string(__name__, 'data/kusto_compute_template.json').decode('ascii'))
synapse_compute_template = json.loads(resource_string(__name__, 'data/synapse_compute_template.json').decode('ascii'))


def get_paginated_compute_results(payload, headers):
    if 'value' not in payload:
        raise ComputeTargetException('Error, invalid paginated response payload, missing "value":\n'
                                     '{}'.format(payload))
    items = payload['value']
    while 'nextLink' in payload:
        next_link = payload['nextLink']

        try:
            resp = ClientBase._execute_func(get_requests_session().get, next_link, headers=headers)
        except requests.Timeout:
            print('Error, request to Machine Learning Compute timed out. Returning with items found so far')
            return items
        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            payload = json.loads(content)
        else:
            raise ComputeTargetException('Received bad response from Machine Learning Compute while retrieving '
                                         'paginated results:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        if 'value' not in payload:
            raise ComputeTargetException('Error, invalid paginated response payload, missing "value":\n'
                                         '{}'.format(payload))
        items += payload['value']

    return items


def get_paginated_compute_supported_vms(payload, headers):
    if 'amlCompute' not in payload:
        raise ComputeTargetException('Error, invalid paginated response payload, missing "amlCompute":\n'
                                     '{}'.format(payload))

    items = []
    required_keys = ['name', 'vCPUs', 'gpus', 'memoryGB', 'maxResourceVolumeMB']
    for i in range(0, len(payload['amlCompute'])):
        for key in required_keys:
            if key not in payload['amlCompute'][i]:
                raise ComputeTargetException('Error, invalid paginated response payload, missing "{}":\n'
                                             '{}'.format(key, payload))
        items.append({key: payload['amlCompute'][i][key] for key in required_keys})

    while 'nextLink' in payload:
        next_link = payload['nextLink']

        try:
            resp = ClientBase._execute_func(get_requests_session().get, next_link, headers=headers)
        except requests.Timeout:
            print('Error, request to Machine Learning Compute timed out. Returning with items found so far')
            return items
        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            payload = json.loads(content)
        else:
            raise ComputeTargetException('Received bad response from Machine Learning Compute while retrieving '
                                         'paginated results:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))

        if 'amlCompute' not in payload:
            raise ComputeTargetException('Error, invalid paginated response payload, missing "amlCompute":\n'
                                         '{}'.format(payload))

        for i in range(0, len(payload['amlCompute'])):
            for key in required_keys:
                if key not in payload['amlCompute'][i]:
                    raise ComputeTargetException('Error, invalid paginated response payload, missing "{}":\n'
                                                 '{}'.format(key, payload))
            items.append({key: payload['amlCompute'][i][key] for key in required_keys})

    return items


def get_paginated_compute_nodes(payload, headers):
    if 'nodes' not in payload:
        raise ComputeTargetException('Error, invalid paginated response payload, missing "nodes":\n'
                                     '{}'.format(payload))
    items = payload['nodes']
    while 'nextLink' in payload:
        next_link = payload['nextLink']

        try:
            resp = ClientBase._execute_func(get_requests_session().get, next_link, headers=headers)
        except requests.Timeout:
            print('Error, request to Machine Learning Compute timed out. Returning with items found so far')
            return items
        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            payload = json.loads(content)
        else:
            raise ComputeTargetException('Received bad response from Machine Learning Compute while retrieving '
                                         'paginated results:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        if 'nodes' not in payload:
            raise ComputeTargetException('Error, invalid paginated response payload, missing "nodes":\n'
                                         '{}'.format(payload))
        items += payload['nodes']

    return items


def convert_seconds_to_duration(duration_in_seconds):
    """
    Convert duration in seconds into ISO-8601 formatted seconds string.

    """
    try:
        duration_in_seconds = int(duration_in_seconds)
    except Exception:
        raise UserErrorException('Invalid input, provide an integer duration in seconds')

    if duration_in_seconds < 0:
        raise UserErrorException('Invalid input, provide duration in seconds')

    return "PT{}S".format(duration_in_seconds)


def convert_duration_to_seconds(attr):
    """
    Convert ISO-8601 formatted duration string into seconds.

    """
    if not attr or attr.isspace():
        return attr
    try:
        duration = isodate.parse_duration(attr)
        return floor(duration.total_seconds())
    except(ValueError, OverflowError, AttributeError):
        raise UserErrorException('Invalid input, provide ISO-8601 formatted duration string')


def get_requests_session():
    """

    :return: A requests.Session object
    :rtype: requests.Session
    """
    _session = requests.Session()

    return _session


def to_json_dict(config):
    """
    Converts dictionary to Json dictiornary by converting keys to CamelCase

    """
    return {to_camel_case(k): v for (k, v) in config.__dict__.items() if not k.startswith("_")}
