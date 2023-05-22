# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains helper methods for dataset service REST APIs."""

import os
import json
from copy import deepcopy

import requests

from azureml.data.constants import _DATASET_TYPE_TABULAR, _DATASET_TYPE_FILE
from azureml.data._loggerfactory import _LoggerFactory
from azureml.data._dataprep_helper import is_dataprep_installed
from azureml.exceptions import UserErrorException
from azureml._base_sdk_common import _ClientSessionId


_FETCH_DATASET_URL = 'data/v1.0/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/' \
                     'workspaces/{}/dataversion/{}/versions/1?includeSavedDatasets=true'


_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


def _dataset_to_dto(dataset, name, description=None, tags=None, dataset_id=None, create_new_version=False):
    dataset_type = _get_type(dataset)
    from azureml._restclient.models.dataset_definition_dto import DatasetDefinitionDto
    from azureml._restclient.models.dataset_state_dto import DatasetStateDto
    from azureml._restclient.models.dataset_dto import DatasetDto

    saved_id = None
    # if create_new_version, then the saved_id must be None, because service doesn't allow the same
    # saved dataset id being registered more than once
    if not create_new_version:
        saved_id = dataset._registration.saved_id if dataset._registration is not None else None
    dataset_definition_dto = DatasetDefinitionDto(
        dataflow=dataset._dataflow.to_json() if is_dataprep_installed() else dataset._definition,
        properties=deepcopy(dataset._properties),
        dataset_definition_state=DatasetStateDto(),
        version_id=str(dataset.version) if dataset.version is not None else None,
        saved_dataset_id=saved_id)

    return DatasetDto(
        name=name,
        dataset_type=dataset_type,
        latest=dataset_definition_dto,
        description=description,
        tags=tags,
        dataset_id=dataset_id,
        is_visible=True)


def _dto_to_registration(workspace, dto):
    from azureml.data._dataset import _DatasetRegistration

    version = _resolve_dataset_version(dto.latest.version_id)
    return _DatasetRegistration(
        workspace=workspace, saved_id=dto.latest.saved_dataset_id, registered_id=dto.dataset_id,
        name=dto.name, version=version, description=dto.description, tags=dto.tags)


def _dto_to_dataset(workspace, dto):
    from azureml._restclient.models.dataset_dto import DatasetDto

    if not isinstance(dto, DatasetDto):
        raise RuntimeError('dto has to be instance of DatasetDto')

    registration = _dto_to_registration(workspace, dto)
    dataflow_json = dto.latest.dataflow
    dataset = None
    if dataflow_json is None or len(dataflow_json) == 0:
        # migrate legacy dataset which has empty dataflow to FileDataset
        data_path = dto.latest.data_path
        if not data_path or 'datastore_name' not in data_path or 'relative_path' not in data_path:
            error = f'Dataset should not have empty dataflow. workspace={workspace}, saved_id={registration.saved_id}'
            _get_logger().error(error)
            raise error

        from azureml.core import Datastore
        from azureml.data.dataset_factory import FileDatasetFactory
        store = Datastore.get(workspace, data_path.datastore_name)
        dataset = FileDatasetFactory.from_files((store, data_path.relative_path))
        dataset._registration = registration

    return _init_dataset(workspace=workspace,
                         dataset_type=dto.dataset_type,
                         dataflow_json=dataflow_json,
                         properties=dto.latest.properties,
                         registration=registration,
                         dataset=dataset)


def _dataset_to_saved_dataset_dto(dataset):
    dataset_type = _get_type(dataset)
    from azureml._restclient.models.saved_dataset_dto import SavedDatasetDto
    return SavedDatasetDto(
        dataset_type=dataset_type,
        properties=deepcopy(dataset._properties),
        dataflow_json=dataset._dataflow.to_json())


def _saved_dataset_dto_to_dataset(workspace, dto):
    from azureml.data._dataset import _DatasetRegistration
    registration = _DatasetRegistration(workspace=workspace, saved_id=dto.id)
    return _init_dataset(workspace=workspace,
                         dataset_type=dto.dataset_type,
                         dataflow_json=dto.dataflow_json,
                         properties=dto.properties,
                         registration=registration)


def _init_dataset(workspace, dataset_type, dataflow_json, properties, registration, dataset=None):
    from azureml.data.tabular_dataset import TabularDataset
    from azureml.data.file_dataset import FileDataset

    if dataset is None:
        dataset_type = _resolve_dataset_type(dataset_type)
        if dataset_type == _DATASET_TYPE_FILE:
            dataset = FileDataset._create(
                definition=dataflow_json,
                properties=properties,
                registration=registration)
        elif dataset_type == _DATASET_TYPE_TABULAR:
            dataset = TabularDataset._create(
                definition=dataflow_json,
                properties=properties,
                registration=registration)
        else:
            raise RuntimeError(f'Unrecognized dataset type "{dataset_type}"')

    fetch_id = 'unregistered-' + registration.saved_id
    try:
        dataset._dataflow._rs_dataflow_yaml = _fetch_rslex_dataflow_yaml(workspace, fetch_id)
    except Exception as e:
        _get_logger().warning(f'Failed to fetch RSLex YAML representation for dataset={fetch_id} '
                              f'from workspace={workspace}=, got error: \'{e}\'')
        dataset._dataflow._rs_dataflow_yaml = None  # still want an attribute even if not fetched successfully
    return dataset


def _resolve_dataset_version(version):
    try:
        return int(version)
    except ValueError:
        _get_logger().warning('Unrecognized dataset version "{}".'.format(version))
        return None


def _resolve_dataset_type(ds_type):
    if ds_type in [_DATASET_TYPE_TABULAR, _DATASET_TYPE_FILE]:
        return ds_type
    if ds_type is not None:
        _get_logger().warning('Unrecognized dataset type "{}".'.format(ds_type))
    # migrate legacy dataset which has dataflow to TabularDataset
    return _DATASET_TYPE_TABULAR


def _get_workspace_uri_path(subscription_id, resource_group, workspace_name):
    return ('/subscriptions/{}/resourceGroups/{}/providers'
            '/Microsoft.MachineLearningServices'
            '/workspaces/{}').format(subscription_id, resource_group, workspace_name)


def _get_type(dataset):
    from azureml.data.tabular_dataset import TabularDataset
    from azureml.data.file_dataset import FileDataset

    if isinstance(dataset, TabularDataset):
        return _DATASET_TYPE_TABULAR
    elif isinstance(dataset, FileDataset):
        return _DATASET_TYPE_FILE
    else:
        raise RuntimeError('Unrecognized dataset type "{}"'.format(type(dataset)))


def _make_request(request_fn,
                  handle_error_fn=None,
                  retries=5,
                  backoff=0.1,
                  status_codes_to_retry=[429, 500, 502, 503, 504]):

    from msrest.exceptions import HttpOperationError
    from azureml._restclient.exceptions import ServiceException
    from time import sleep

    cur_attempt, _delay = 0, 0
    while cur_attempt <= retries:
        try:
            cur_attempt += 1
            return (True, request_fn())
        except HttpOperationError as error:
            try:
                status_code = error.response.status_code
                if handle_error_fn:
                    # request specific error handling, no retries if the error is handled
                    handled_error = handle_error_fn(error)
                    if handled_error:
                        return (False, handled_error)

                if status_code in status_codes_to_retry:
                    if cur_attempt <= retries:
                        _get_logger().warning('Request `{}` failed at attempt {} with status code: {}.'
                                              'Retrying in {} seconds'.format(
                                                  request_fn.__name__,
                                                  cur_attempt,
                                                  status_code,
                                                  _delay))

                        sleep(_delay)
                        _delay = backoff * (2 ** (cur_attempt - 1))
                        continue
                    else:
                        message = 'Request {} failed after {} attempts, last status code was {}.'
                        'No retries left.'.format(
                            request_fn.__name__, cur_attempt, status_code)
                        print(message)
                        _get_logger().error(message)
                if status_code >= 400 and status_code < 500:
                    # generic user error handling
                    message = ""
                    try:
                        message = json.loads(error.response.content)['error']['message']
                    except Exception:
                        pass
                    error = UserErrorException('Request failed ({}): {}'.format(status_code, message))
                else:
                    # mapping service request failure to ServiceException
                    _get_logger().error('Request failed with {}: {}'.format(status_code, error.message))
                    error = ServiceException(error)
            except Exception as exp:
                _get_logger().error('Exception while handling request error: {}'.format(repr(exp)))
            return (False, error)
        except KeyError as key_error:
            _get_logger().error('Request failed with: {} at attempt {}'.format(key_error, cur_attempt))
            # Due to this bug  https://msdata.visualstudio.com/Vienna/_workitems/edit/1657452/
            # The issue might be caused by data race, however, we don't know where is the code to change the os.envrion
            # maybe other packages, add retry here to avoid entire job get failed
            if cur_attempt <= retries:
                continue
            return (False, key_error)
        except Exception as other_exception:
            _get_logger().error('Request failed with: {}'.format(other_exception))
            return (False, other_exception)


def _restclient(ws):
    host_env = os.environ.get('AZUREML_SERVICE_ENDPOINT')
    auth = ws._auth

    from azureml._base_sdk_common.service_discovery import get_service_url
    from msrest.authentication import BasicTokenAuthentication
    from azureml._restclient.rest_client import RestClient

    host = host_env or get_service_url(
        auth,
        _get_workspace_uri_path(
            ws._subscription_id,
            ws._resource_group,
            ws._workspace_name),
        ws._workspace_id,
        ws.discovery_url)

    auth_header = ws._auth.get_authentication_header()['Authorization']
    access_token = auth_header[7:]  # 7 == len('Bearer ')

    return RestClient(base_url=host, credentials=BasicTokenAuthentication({
        'access_token': access_token
    }))


def _fetch_rslex_dataflow_yaml(workspace, fetch_id):
    if fetch_id is None:
        raise ValueError('Require a Dataset name or saved ID to fetch RSlex Dataflow YAML representation')

    rest_client = _restclient(workspace)
    request_url = rest_client.config.base_url + '/' + _FETCH_DATASET_URL.format(workspace.subscription_id,
                                                                                workspace.resource_group,
                                                                                workspace.name,
                                                                                fetch_id)
    access_header = workspace._auth.get_authentication_header()
    response = requests.get(request_url, headers=access_header)
    if response.status_code == 200:
        return response.json()['legacyDataflow']
    raise response.raise_for_status()


_custom_headers = {'x-ms-client-session-id': _ClientSessionId}
