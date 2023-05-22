# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing resources configuration for Azure Machine Learning entities."""

import logging
from azureml.exceptions import WebserviceException

module_logger = logging.getLogger(__name__)


class ResourceConfiguration(object):
    """Defines the details for the resource configuration of Azure Machine Learning resources.

    .. remarks::

        Initialize a resource configuration with this class. For example, the following code shows
        how to register a model specifying framework, input and output datasets, and resource configuration.

        .. code-block:: python

            import sklearn

            from azureml.core import Model
            from azureml.core.resource_configuration import ResourceConfiguration


            model = Model.register(workspace=ws,
                                   model_name='my-sklearn-model',                # Name of the registered model in your workspace.
                                   model_path='./sklearn_regression_model.pkl',  # Local file to upload and register as a model.
                                   model_framework=Model.Framework.SCIKITLEARN,  # Framework used to create the model.
                                   model_framework_version=sklearn.__version__,  # Version of scikit-learn used to create the model.
                                   sample_input_dataset=input_dataset,
                                   sample_output_dataset=output_dataset,
                                   resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                                   description='Ridge regression model to predict diabetes progression.',
                                   tags={'area': 'diabetes', 'type': 'regression'})

            print('Name:', model.name)
            print('Version:', model.version)

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/deploy-to-cloud/model-register-and-deploy.ipynb


    :param cpu: The number of CPU cores to allocate for this resource. Can be a decimal.
    :type cpu: float
    :param memory_in_gb: The amount of memory (in GB) to allocate for this resource. Can be a decimal.
    :type memory_in_gb: float
    :param gpu: The number of GPUs to allocate for this resource.
    :type gpu: int
    """

    _expected_payload_keys = ['cpu', 'memoryInGB', 'gpu']

    def __init__(self, cpu=None, memory_in_gb=None, gpu=None):
        """Initialize the  ResourceConfiguration.

        :param cpu: The number of CPU cores to allocate for this resource. Can be a decimal.
        :type cpu: float
        :param memory_in_gb: The amount of memory (in GB) to allocate for this resource. Can be a decimal.
        :type memory_in_gb: float
        :param gpu: The number of GPUs to allocate for this resource.
        :type gpu: int
        """
        self.cpu = cpu
        self.memory_in_gb = memory_in_gb
        self.gpu = gpu

    def serialize(self):
        """Convert this ResourceConfiguration into a JSON serialized dictionary.

        :return: The JSON representation of this ResourceConfiguration.
        :rtype: dict
        """
        return {'cpu': self.cpu, 'memoryInGB': self.memory_in_gb, 'gpu': self.gpu}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a ResourceConfiguration object.

        :param payload_obj: A JSON object to convert to a ResourceConfiguration object.
        :type payload_obj: dict
        :return: The ResourceConfiguration representation of the provided JSON object.
        :rtype: azureml.core.resource_configuration.ResourceConfiguration
        """
        if payload_obj is None:
            return None
        for payload_key in ResourceConfiguration._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for ResourceConfiguration:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return ResourceConfiguration(payload_obj['cpu'], payload_obj['memoryInGB'], payload_obj['gpu'])

    def _validate_configuration(self):
        """Check that the specified configuration values are valid.

        Will raise a :class:`azureml.exceptions.WebserviceException` if validation fails.

        :raises: azureml.exceptions.WebserviceException
        """
        error = ""
        if self.cpu and self.cpu <= 0:
            error += 'Invalid configuration, cpu must be greater than zero.\n'
        if self.memory_in_gb and self.memory_in_gb <= 0:
            error += 'Invalid configuration, memory_in_gb must be greater than zero.\n'
        if self.gpu and not isinstance(self.gpu, int) and self.gpu <= 0:
            error += 'Invalid configuration, gpu must be integer and greater than zero.\n'

        if error:
            raise WebserviceException(error, logger=module_logger)

    @staticmethod
    def _from_dto(container_resource_configuration):
        """Convert an autorest ContainerResourceConfiguration object into a ResourceConfiguration object.

        :param payload_obj: A ContainerResourceConfiguration to convert to a ResourceConfiguration object
        :type payload_obj: azureml._restclient.models.ContainerResourceRequirements
        :return: The ResourceConfiguration representation of the provided json object
        :rtype: azureml.core.resource_configuration.ResourceConfiguration
        """
        if container_resource_configuration is None:
            return None

        return ResourceConfiguration(cpu=container_resource_configuration.cpu,
                                     memory_in_gb=container_resource_configuration.memory_in_gb,
                                     gpu=container_resource_configuration.gpu)
