# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for profiling models in Azure Machine Learning."""

import logging
import sys
import time
import json
import warnings
from dateutil.parser import parse

try:
    from abc import ABCMeta
    ABC = ABCMeta('ABC', (), {})
except ImportError:
    from abc import ABC
from abc import abstractmethod

from azureml._model_management._constants import (
    PROFILE_FAILURE_MESSAGE,
    PROFILE_PARTIAL_SUCCESS_MESSAGE,
)
from azureml._model_management._util import get_mms_operation, get_operation_output
from azureml.exceptions import WebserviceException, UserErrorException

module_logger = logging.getLogger(__name__)


MIN_PROFILE_CPU = 0.1
MAX_PROFILE_CPU = 3.5
MIN_PROFILE_MEMORY = 0.1
MAX_PROFILE_MEMORY = 15.0


class _ModelEvaluationResultBase(ABC):
    """_ModelEvaluationResultBase abstract object that serves as a base for results of profiling and validation."""

    @property
    @classmethod
    @abstractmethod
    def _model_eval_type(cls):
        return NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def _general_mms_suffix(cls):
        return NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def _expected_payload_keys(cls):
        return NotImplementedError

    @abstractmethod
    def __repr__(self):
        """Return the string representation of the _ModelEvaluationResultBase object.

        :return: String representation of the _ModelEvaluationResultBase object
        :rtype: str
        """
        return NotImplementedError

    @abstractmethod
    def get_details(self):
        """Return the the observed metrics and other details of the model test operation.

        :return: Dictionary of metrics
        :rtype: dict[str, float]
        """
        return NotImplementedError

    _details_keys_success = [
        'requestedCpu',
        'requestedMemoryInGB',
        'requestedQueriesPerSecond',
        'maxUtilizedMemoryInGB',
        'maxUtilizedCpu',
        'totalQueries',
        'successQueries',
        'successRate',
        'averageLatencyInMs',
        'latencyPercentile50InMs',
        'latencyPercentile90InMs',
        'latencyPercentile95InMs',
        'latencyPercentile99InMs',
        'latencyPercentile999InMs',
        'measuredQueriesPerSecond',
        'state',
        'name',
        'description',
        'createdTime',
        'error'
    ]

    _details_keys_error = [
        'name',
        'description',
        'state',
        'requestedCpu',
        'requestedMemoryInGB',
        'requestedQueriesPerSecond',
        'error',
        'errorLogsUri',
        'createdTime'
    ]

    def __init__(self, workspace, name):
        """Initialize the _ModelEvaluationResultBase object.

        :param workspace: The workspace object containing the model.
        :type workspace: azureml.core.Workspace
        :param name: The name of the profile to construct and retrieve.
        :type name: str
        :rtype: azureml.core.profile._ModelEvaluationResultBase
        """
        self.workspace = workspace
        self.name = name
        # setting default values for properties needed to communicate with MMS
        self.create_operation_id = None
        self._model_test_result_suffix = None
        self._model_test_result_params = {}

        super(_ModelEvaluationResultBase, self).__init__()

    def _initialize(self, obj_dict):
        """Initialize the base properites of an instance of subtype of _ModelEvaluationResultBase.

        This is used because the constructor is used as a getter.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        self.name = obj_dict.get('name')
        self.description = obj_dict.get('description')
        self.created_time = (
            parse(obj_dict['createdTime']) if 'createdTime' in obj_dict else None
        )
        self.id = obj_dict.get('id')
        container_resource_requirements = obj_dict.get('containerResourceRequirements')
        self.requested_cpu = (
            container_resource_requirements['cpu']
            if container_resource_requirements
            else None
        )
        self.requested_memory_in_gb = (
            container_resource_requirements['memoryInGB']
            if container_resource_requirements
            else None
        )
        self.requested_queries_per_second = obj_dict.get('requestedQueriesPerSecond')
        self.input_dataset_id = obj_dict.get('inputDatasetId')
        self.state = obj_dict.get('state')
        self.model_ids = obj_dict.get('modelIds')
        # detail properties
        self.max_utilized_memory = obj_dict.get('maxUtilizedMemoryInGB')
        self.max_utilized_cpu = obj_dict.get('maxUtilizedCpu')
        self.measured_queries_per_second = obj_dict.get('measuredQueriesPerSecond')
        self.environment = obj_dict.get('environment')
        self.error = obj_dict.get('error')
        self.error_logs_url = obj_dict.get('errorLogsUri')
        self.total_queries = obj_dict.get('totalQueries')
        self.success_queries = obj_dict.get('successQueries')
        self.success_rate = obj_dict.get('successRate')
        self.average_latency_in_ms = obj_dict.get('averageLatencyInMs')
        self.latency_percentile_50_in_ms = obj_dict.get('latencyPercentile50InMs')
        self.latency_percentile_90_in_ms = obj_dict.get('latencyPercentile90InMs')
        self.latency_percentile_95_in_ms = obj_dict.get('latencyPercentile95InMs')
        self.latency_percentile_99_in_ms = obj_dict.get('latencyPercentile99InMs')
        self.latency_percentile_999_in_ms = obj_dict.get('latencyPercentile999InMs')

    @classmethod
    def _validate_get_payload(cls, payload):
        """Validate the returned _ModelEvaluationResultBase payload.

        :param payload:
        :type payload: dict
        :return:
        :rtype: None
        """
        for payload_key in cls._expected_payload_keys:
            if payload_key not in payload:
                raise WebserviceException(
                    'Invalid payload for %s, missing %s:\n %s'
                    % (cls._model_eval_type, payload_key, payload)
                )

    def _get_operation_state(self):
        """Get the current async operation state for the a model test operation.

        :return:
        :rtype: (str, dict)
        """
        resp_content = get_mms_operation(self.workspace, self.create_operation_id)
        state = resp_content['state']
        error = resp_content['error'] if 'error' in resp_content else None
        async_operation_request_id = resp_content['parentRequestId']
        return state, error, async_operation_request_id

    def _update_creation_state(self):
        """Refresh the current state of the in-memory object.

        Perform an in-place update of the properties of the object based on the current state of the
        corresponding cloud object. Primarily useful for manual polling of creation state.

        :raises: azureml.exceptions.WebserviceException
        """
        resp = get_operation_output(self.workspace, self._model_test_result_suffix, self._model_test_result_params)
        if resp.status_code != 200:
            error_message = 'Model {} result with name {}'.format(
                self.__class__._model_eval_type, self.name
            )
            error_message += ' not found in provided workspace'
            raise WebserviceException(error_message)
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        model_test_json = json.loads(content)['value'][0]
        self._validate_get_payload(model_test_json)
        self._initialize(model_test_json)

    @abstractmethod
    def wait_for_completion(self, show_output=False):
        """Wait for the model evaluation process to finish.

        :param show_output: Boolean option to print more verbose output. Defaults to False.
        :type show_output: bool
        """
        if not (self.workspace and self.create_operation_id):
            raise UserErrorException('wait_for_completion operation cannot be performed on this object.'
                                     'Make sure the object was created via the appropriate method '
                                     'in the Model class')
        operation_state, error, request_id = self._get_operation_state()
        self.parent_request_id = request_id
        current_state = operation_state
        if show_output:
            sys.stdout.write('{}'.format(current_state))
            sys.stdout.flush()
        while operation_state not in ['Cancelled', 'Succeeded', 'Failed', 'TimedOut']:
            time.sleep(5)
            operation_state, error, _ = self._get_operation_state()
            if show_output:
                sys.stdout.write('.')
                if operation_state != current_state:
                    sys.stdout.write('\n{}'.format(operation_state))
                    current_state = operation_state
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        module_logger.info(
            'Model {} operation with name {} finished operation {}\n'.format(
                self.__class__._model_eval_type, self.name, operation_state
            )
        )
        if operation_state == 'Failed':
            if error and 'statusCode' in error and 'message' in error:
                module_logger.info(
                    'Model {} failed with\n'
                    'StatusCode: {}\n'
                    'Message: {}\n'
                    'Operation ID: {}\n'
                    'Request ID: {}\n'.format(
                        self.__class__._model_eval_type,
                        error['statusCode'],
                        error['message'],
                        self.create_operation_id,
                        self.parent_request_id
                    )
                )
            else:
                module_logger.info(
                    'Model profiling failed, unexpected error response:\n'
                    '{}\n'
                    'Operation ID: {}\n'
                    'Request ID: {}\n'.format(
                        error,
                        self.create_operation_id,
                        self.parent_request_id)
                )
        self._update_creation_state()

    def serialize(self):
        """Convert this _ModelEvaluationResultBase object into a json serialized dictionary.

        :return: The json representation of this _ModelEvaluationResultBase
        :rtype: dict
        """
        created_time = self.created_time.isoformat() if self.created_time else None
        return {
            'id': self.id,
            'name': self.name,
            'createdTime': created_time,
            'state': self.state,
            'description': self.description,
            'requestedCpu': self.requested_cpu,
            'requestedMemoryInGB': self.requested_memory_in_gb,
            'requestedQueriesPerSecond': self.requested_queries_per_second,
            'inputDatasetId': self.input_dataset_id,
            'maxUtilizedMemoryInGB': self.max_utilized_memory,
            'totalQueries': self.total_queries,
            'successQueries': self.success_queries,
            'successRate': self.success_rate,
            'averageLatencyInMs': self.average_latency_in_ms,
            'latencyPercentile50InMs': self.latency_percentile_50_in_ms,
            'latencyPercentile90InMs': self.latency_percentile_90_in_ms,
            'latencyPercentile95InMs': self.latency_percentile_95_in_ms,
            'latencyPercentile99InMs': self.latency_percentile_99_in_ms,
            'latencyPercentile999InMs': self.latency_percentile_999_in_ms,
            'modelIds': self.model_ids,
            'environment': self.environment,
            'maxUtilizedCpu': self.max_utilized_cpu,
            'measuredQueriesPerSecond': self.measured_queries_per_second,
            'error': self.error,
            'errorLogsUri': self.error_logs_url
        }


class ModelProfile(_ModelEvaluationResultBase):
    """
    Contains the results of a profiling run.

    A model profile of a model is a resource requirement recommendation. A ModelProfile object is returned from
    the :meth:`azureml.core.model.Model.profile` method of the :class:`azureml.core.model.Model` class.

    .. remarks::

        The following example shows how to return a ModelProfile object.

        .. code-block:: python

            profile = Model.profile(ws, "profilename", [model], inference_config, input_dataset=dataset)
            profile.wait_for_profiling(True)
            profiling_details = profile.get_details()
            print(profiling_details)

    :param workspace: The workspace object containing the model.
    :type workspace: azureml.core.Workspace
    :param name: The name of the profile to create and retrieve.
    :type name: str
    :rtype: azureml.core.profile.ModelProfile
    :raises: azureml.exceptions.WebserviceException
    """

    _model_eval_type = 'profiling'
    _general_mms_suffix = '/profiles'
    _expected_payload_keys = [
        'name',
        'description',
        'id',
        'state',
        'inputDatasetId',
        'containerResourceRequirements',
        'requestedQueriesPerSecond',
        'createdTime',
        'modelIds'
    ]

    _profile_recommended_cpu_key = 'recommendedCpu'
    _profile_recommended_memory_key = 'recommendedMemoryInGB'

    _details_keys_success = _ModelEvaluationResultBase._details_keys_success + [
        _profile_recommended_memory_key,
        _profile_recommended_cpu_key
    ]

    def __init__(self, workspace, name):
        """Initialize the ModelProfile object.

        :param workspace: The workspace object containing the model.
        :type workspace: azureml.core.Workspace
        :param name: The name of the profile to create and retrieve.
        :type name: str
        :rtype: azureml.core.profile.ModelProfile
        :raises: azureml.exceptions.WebserviceException
        """
        super(ModelProfile, self).__init__(workspace, name)

        if workspace and name:
            self._model_test_result_suffix = self._general_mms_suffix
            self._model_test_result_params = {'name': name}
            # retrieve object from MMS and update state
            self._update_creation_state()
        else:
            # sets all properties associated with profiling result to None
            self._initialize({})

    def _initialize(self, obj_dict):
        """Initialize the Profile instance.

        This is used because the constructor is used as a getter.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        super(ModelProfile, self)._initialize(obj_dict)
        self.recommended_memory = obj_dict.get(self._profile_recommended_memory_key)
        self.recommended_cpu = obj_dict.get(self._profile_recommended_cpu_key)

    def wait_for_completion(self, show_output=False):
        """Wait for the model to finish profiling.

        :param show_output: Boolean option to print more verbose output. Defaults to False.
        :type show_output: bool
        """
        super().wait_for_completion(show_output)
        if self.state == 'Failed':
            warnings.warn(PROFILE_FAILURE_MESSAGE % (self.error['message'], self.parent_request_id),
                          category=UserWarning, stacklevel=2)
        elif self.error:
            warnings.warn(PROFILE_PARTIAL_SUCCESS_MESSAGE % self.error['message'],
                          category=UserWarning, stacklevel=2)

    def serialize(self):
        """Convert this Profile into a JSON serialized dictionary.

        :return: The JSON representation of this Profile
        :rtype: dict
        """
        dict_repr = super(ModelProfile, self).serialize()
        dict_repr.update(
            {
                self._profile_recommended_memory_key: self.recommended_memory,
                self._profile_recommended_cpu_key: self.recommended_cpu
            }
        )
        return dict_repr

    def __repr__(self):
        """Return the string representation of the ModelProfile object.

        :return: String representation of the ModelProfile object
        :rtype: str
        """
        str_repr = []
        str_repr.append(('workspace' + '=%s') % repr(self.workspace))
        for key in self.__dict__:
            if key[0] != '_' and key not in ['workspace']:
                str_repr.append((key + '=%s') % self.__dict__[key])
        str_repr = '%s(%s)' % (self.__class__.__name__, ', '.join(str_repr))
        return str_repr

    def get_details(self):
        """Get the details of the profiling result.

        Return the the observed metrics (various latency percentiles, maximum utilized cpu and memory, etc.)
        and recommended resource requirements in case of a success.

        :return: A dictionary of recommended resource requirements.
        :rtype: dict[str, float]
        """
        dict_repr = self.serialize()
        if dict_repr['state'] == 'Succeeded':
            success_repr = {
                k: dict_repr[k]
                for k in dict_repr
                if (
                    dict_repr[k] is not None
                    and k in self.__class__._details_keys_success
                )
            }
            if 'error' in success_repr:
                warnings.warn(
                    PROFILE_PARTIAL_SUCCESS_MESSAGE % success_repr['error']['message'],
                    category=UserWarning, stacklevel=2)
            return success_repr
        return {
            k: dict_repr[k]
            for k in dict_repr
            if (dict_repr[k] is not None and k in self.__class__._details_keys_error)
        }
