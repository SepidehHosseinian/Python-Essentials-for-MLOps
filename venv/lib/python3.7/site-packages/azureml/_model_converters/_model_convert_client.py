# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
""" Call model convert, wait for model convert operation and retrieve the converted model """

import json
import requests
from azureml.exceptions import WebserviceException
from azureml._model_management._constants import MMS_WORKSPACE_API_VERSION
from azureml._model_management._constants import MMS_SYNC_TIMEOUT_SECONDS
from azureml._model_management._util import _get_mms_url


class ModelConvertClient(object):
    def __init__(self, workspace):
        """
        :param workspace:
        :type workspace: azureml.core.workspace.Workspace
        :return:
        :rtype: None
        """

        self._workspace = None
        self._mms_endpoint = None

        if workspace:
            self._workspace = workspace
            self._mms_endpoint = _get_mms_url(workspace)

        else:
            raise Exception('Workspace must be provided.')

    def convert_model(self, request):
        """
        :param request:
        :type request: ModelConvertRequest
        :return: submit model conversion and return opertion id
        :rtype: str
        """

        headers = {'Content-Type': 'application/json'}
        headers.update(self._workspace._auth.get_authentication_header())
        params = {'api-version': MMS_WORKSPACE_API_VERSION}

        if request.model_id:
            model_endpoint = self._mms_endpoint + '/models' + '/{}/convert'.format(request.model_id)
        else:
            raise Exception('Model id is missing in the request {}.'.format(request))

        json_payload = json.loads(request.toJSON())

        try:
            resp = requests.post(model_endpoint, params=params, headers=headers, json=json_payload)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise Exception('Error connecting to {}.'.format(model_endpoint))
        except requests.exceptions.HTTPError:
            raise Exception('Received bad response from Model Management Service:\n'
                            'Response Code: {}\n'
                            'Headers: {}\n'
                            'Content: {}'.format(resp.status_code, resp.headers, resp.content))

        if resp.status_code != 202:
            raise WebserviceException('Error occurred converting model:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content))

        if 'Operation-Location' in resp.headers:
            operation_location = resp.headers['Operation-Location']
        else:
            raise WebserviceException('Missing response header key: Operation-Location')

        operation_status_id = operation_location.split('/')[-1]

        return operation_status_id

    def get_operation_state(self, operation_id):
        """
        :param operation_id:
        :type operation_id: str
        :return: operation state, operation content
        :rtype: (str, object)
        """
        headers = {'Content-Type': 'application/json'}
        headers.update(self._workspace._auth.get_authentication_header())
        params = {'api-version': MMS_WORKSPACE_API_VERSION}

        operation_endpoint = self._mms_endpoint + '/operations/{}'.format(operation_id)
        resp = requests.get(operation_endpoint, headers=headers, params=params, timeout=MMS_SYNC_TIMEOUT_SECONDS)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Resource Provider:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = json.loads(content)
        state = content['state']
        return state, content
