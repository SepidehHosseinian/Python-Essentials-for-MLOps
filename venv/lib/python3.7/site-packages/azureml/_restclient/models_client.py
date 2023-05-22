# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Access ModelsClient"""

import logging

from azureml._base_sdk_common import _ClientSessionId
from azureml._base_sdk_common.user_agent import get_user_agent
from azureml._model_management._util import populate_model_not_found_details
from azureml._restclient.models import Model, ModelErrorResponseException
from azureml.exceptions import WebserviceException

from .workspace_client import WorkspaceClient

module_logger = logging.getLogger(__name__)

MODELS_SERVICE_VERSION = "2018-11-19"
RETRY_LIMIT = 3
BACKOFF_START = 2
DEFAULT_MIMETYPE = "application/json"


class ModelsClient(WorkspaceClient):
    """Model client class"""

    def error_with_model_id_handling(func):
        def wrapper(self, model_id, *args, **kwargs):
            try:
                return func(self, model_id, *args, **kwargs)
            except ModelErrorResponseException as ex:
                return ModelsClient.model_error_response_to_webservice_exception(ex, id=model_id)

        return wrapper

    def error_with_model_param_handling(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ModelErrorResponseException as ex:
                return ModelsClient.model_error_response_to_webservice_exception(ex, **kwargs)

        return wrapper

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_modelmanagement_restclient(user_agent=user_agent)

    @error_with_model_param_handling
    def register_model(self, name, tags=None, properties=None, description=None, url=None, mime_type=DEFAULT_MIMETYPE,
                       framework=None, framework_version=None, unpack=False, experiment_name=None, run_id=None,
                       datasets=None, sample_input_data=None, sample_output_data=None, resource_requirements=None):
        model = Model(name=name,
                      kv_tags=tags,
                      properties=properties,
                      description=description,
                      url=url,
                      mime_type=mime_type,
                      framework=framework,
                      framework_version=framework_version,
                      unpack=unpack,
                      experiment_name=experiment_name,
                      run_id=run_id,
                      datasets=datasets,
                      sample_input_data=sample_input_data,
                      sample_output_data=sample_output_data,
                      resource_requirements=resource_requirements)

        return self.\
            _execute_with_workspace_arguments(self._client.ml_models.register, model,
                                              custom_headers=ModelsClient.get_modelmanagement_custom_headers())

    @error_with_model_id_handling
    def get_by_id(self, model_id):
        return self.\
            _execute_with_workspace_arguments(self._client.ml_models.query_by_id, model_id,
                                              custom_headers=ModelsClient.get_modelmanagement_custom_headers())

    @error_with_model_param_handling
    def query(self, name=None, tag=None, framework=None, description=None, count=None, skip_token=None, tags=None,
              properties=None, run_id=None, dataset_id=None, order_by=None, latest_version_only=None):

        return self.\
            _execute_with_workspace_arguments(self._client.ml_models.list_query, name=name, tag=tag,
                                              framework=framework, description=description, count=count,
                                              skip_token=skip_token, tags=tags, properties=properties,
                                              run_id=run_id, dataset_id=dataset_id, order_by=order_by,
                                              latest_version_only=latest_version_only, is_paginated=True,
                                              custom_headers=ModelsClient.get_modelmanagement_custom_headers())

    @error_with_model_id_handling
    def patch(self, model_id, body=None):
        return self.\
            _execute_with_workspace_arguments(self._client.ml_models.patch, id=model_id, body=body,
                                              custom_headers=ModelsClient.get_modelmanagement_custom_headers())

    @error_with_model_id_handling
    def delete(self, model_id):
        return self.\
            _execute_with_workspace_arguments(self._client.ml_models.delete, id=model_id,
                                              custom_headers=ModelsClient.get_modelmanagement_custom_headers())

    @staticmethod
    def model_error_response_to_webservice_exception(ex, **kwargs):
        if not isinstance(ex, ModelErrorResponseException):
            raise WebserviceException('Received unexpected exception: {}'.format(ex), logger=module_logger)

        if ex.error.status_code == 404:
            error_message = populate_model_not_found_details(**kwargs)

            raise WebserviceException(error_message, logger=module_logger)

        raise WebserviceException('Received bad response from Model Management Service:\n'
                                  'Response Code: {}\n'
                                  'Correlation: {}\n'
                                  'Content: {}'.format(ex.error.status_code, ex.error.correlation, ex.error.details),
                                  logger=module_logger)

    @staticmethod
    def get_modelmanagement_custom_headers():
        return {'User-Agent': get_user_agent(), 'x-ms-client-session-id': _ClientSessionId}
