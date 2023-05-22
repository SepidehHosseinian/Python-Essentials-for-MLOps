# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Module for describing Container Resource Requirements in Azure Machine Learning."""


import logging
from azureml.exceptions import WebserviceException
module_logger = logging.getLogger(__name__)


class ContainerResourceRequirements(object):
    """Defines the resource requirements for a container used by the Webservice.

    To specify autoscaling configuration, you will typically use the ``deploy_configuration``
    method of the :class:`azureml.core.webservice.aks.AksWebservice` class or the
    :class:`azureml.core.webservice.aci.AciWebservice` class.

    :param cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
    :type cpu: float
    :param memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
    :type memory_in_gb: float
    :param cpu_limit: The max number of CPU cores this Webservice is allowed to use. Can be a decimal.
    :type cpu_limit: float
    :param memory_in_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :type memory_in_gb_limit: float
    :param gpu: The number of GPU cores to allocate for this Webservice.
    :type gpu: int
    """

    _expected_payload_keys = ['cpu', 'cpuLimit', 'memoryInGB', 'memoryInGBLimit', 'gpu']

    def __init__(self, cpu, memory_in_gb, gpu=None, cpu_limit=None, memory_in_gb_limit=None):
        """Initialize the container resource requirements.

        :param cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
        :type cpu: float
        :param memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        :type memory_in_gb: float
        :param cpu_limit: The max number of CPU cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_limit: float
        :param memory_in_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use.
                                    Can be a decimal.
        :type memory_in_gb_limit: float
        :param gpu: The number of GPU cores to allocate for this Webservice.
        :type gpu: int
        """
        self.cpu = cpu
        self.cpu_limit = cpu_limit
        self.memory_in_gb = memory_in_gb
        self.memory_in_gb_limit = memory_in_gb_limit
        self.gpu = gpu

    def serialize(self):
        """Convert this ContainerResourceRequirements object into a JSON serialized dictionary.

        :return: The JSON representation of this ContainerResourceRequirements.
        :rtype: dict
        """
        return {'cpu': self.cpu, 'cpuLimit': self.cpu_limit,
                'memoryInGB': self.memory_in_gb, 'memoryInGBLimit': self.memory_in_gb_limit,
                'gpu': self.gpu}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a ContainerResourceRequirements object.

        :param payload_obj: A JSON object to convert to a ContainerResourceRequirements object.
        :type payload_obj: dict
        :return: The ContainerResourceRequirements representation of the provided JSON object.
        :rtype: azureml.core.webservice.webservice.ContainerResourceRequirements
        """
        for payload_key in ContainerResourceRequirements._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for ContainerResourceRequirements:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return ContainerResourceRequirements(cpu=payload_obj['cpu'], memory_in_gb=payload_obj['memoryInGB'],
                                             gpu=payload_obj['gpu'], cpu_limit=payload_obj['cpuLimit'],
                                             memory_in_gb_limit=payload_obj['memoryInGBLimit'])
