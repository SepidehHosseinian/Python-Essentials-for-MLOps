# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for deploying machine learning models as web service endpoints on Azure Kubernetes Service.

Azure Kubernetes Service (AKS) is recommended for scenarios where you need full container orchestration,
including service discovery across multiple containers, automatic scaling, and coordinated application upgrades.

For more information, see [Deploy a model to Azure Kubernetes
Service](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service).
"""

import copy
import json
import logging
import re
import requests
import warnings
from dateutil.parser import parse
from azureml._base_sdk_common.tracking import global_tracking_info_registry
from azureml._model_management._constants import AKS_ENDPOINT_TYPE
from azureml._model_management._constants import AKS_WEBSERVICE_TYPE
from azureml._model_management._constants import MMS_MIGRATION_TIMEOUT_SECONDS
from azureml._model_management._constants import NAMESPACE_REGEX
from azureml._model_management._constants import WEBSERVICE_SWAGGER_PATH
from azureml._model_management._constants import SERVICE_REQUEST_OPERATION_UPDATE
from azureml._model_management._util import _get_mms_url
from azureml._model_management._util import build_and_validate_no_code_environment_image_request
from azureml._model_management._util import convert_parts_to_environment
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import webservice_name_validation
from azureml._restclient.clientbase import ClientBase
from azureml.core.compute import ComputeTarget
from azureml.core.environment import Environment
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.webservice.webservice import WebserviceDeploymentConfiguration, WebServiceAccessToken
from azureml.exceptions import WebserviceException
from enum import Enum

module_logger = logging.getLogger(__name__)


class AksWebservice(Webservice):
    """Represents a machine learning model deployed as a web service endpoint on Azure Kubernetes Service.

    A deployed service is created from a model, script, and associated files. The resulting web
    service is a load-balanced, HTTP endpoint with a REST API. You can send data to this API and
    receive the prediction returned by the model.

    AksWebservice deploys a single service to one endpoint. To deploy multiple services to one endpoint, use the
    :class:`azureml.core.webservice.AksEndpoint` class.

    For more information, see `Deploy a model to an Azure Kubernetes Service
    cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.

    .. remarks::

        The recommended deployment pattern is to create a deployment configuration object with the
        ``deploy_configuration`` method and then use it with the ``deploy`` method of the
        :class:`azureml.core.model.Model` class as shown below.

        .. code-block:: python

            # Set the web service configuration (using default here)
            aks_config = AksWebservice.deploy_configuration()

            # # Enable token auth and disable (key) auth on the webservice
            # aks_config = AksWebservice.deploy_configuration(token_auth_enabled=True, auth_enabled=False)

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/production-deploy-to-aks/production-deploy-to-aks.ipynb


        There are a number of ways to deploy a model as a webservice, including with the:

        * ``deploy`` method of the :class:`azureml.core.model.Model` for models already registered in the workspace.

        * ``deploy_from_image`` method of :class:`azureml.core.webservice.Webservice`.

        * ``deploy_from_model`` method of :class:`azureml.core.webservice.Webservice` for models already registered
          in the workspace. This method will create an image.

        * ``deploy`` method of the :class:`azureml.core.webservice.Webservice`, which will register a model and
          create an image.

        For information on working with webservices, see

        * `Consume an Azure Machine Learning model deployed
          as a web service <https://docs.microsoft.com/azure/machine-learning/how-to-consume-web-service>`_

        * `Monitor and collect data from ML web service
          endpoints <https://docs.microsoft.com/azure/machine-learning/how-to-enable-app-insights>`_

        * `Troubleshooting
          deployment <https://docs.microsoft.com/azure/machine-learning/how-to-troubleshoot-deployment>`_

        The *Variables* section lists attributes of a local representation of the cloud AksWebservice object. These
        variables should be considered read-only. Changing their values will not be reflected in the corresponding
        cloud object.

    :var enable_app_insights: Whether or not AppInsights logging is enabled for the Webservice.
    :vartype enable_app_insights: bool
    :var autoscaler: The Autoscaler object for the Webservice.
    :vartype autoscaler: azureml.core.webservice.webservice.AutoScaler
    :var compute_name: The name of the ComputeTarget that the Webservice is deployed to.
    :vartype compute_name: str
    :var container_resource_requirements: The container resource requirements for the Webservice.
    :vartype container_resource_requirements: azureml.core.webservice.aks.ContainerResourceRequirements
    :var liveness_probe_requirements: The liveness probe requirements for the Webservice.
    :vartype liveness_probe_requirements: azureml.core.webservice.aks.LivenessProbeRequirements
    :var data_collection: The DataCollection object for the Webservice.
    :vartype data_collection: azureml.core.webservice.aks.DataCollection
    :var max_concurrent_requests_per_container: The maximum number of concurrent requests per container for
        the Webservice.
    :vartype max_concurrent_requests_per_container: int
    :var max_request_wait_time: The maximum request wait time for the Webservice, in milliseconds.
    :vartype max_request_wait_time: int
    :var num_replicas: The number of replicas for the Webservice. Each replica corresponds to an AKS pod.
    :vartype num_replicas: int
    :var scoring_timeout_ms: The scoring timeout for the Webservice, in milliseconds.
    :vartype scoring_timeout_ms: int
    :var azureml.core.webservice.AksWebservice.scoring_uri: The scoring endpoint for the Webservice
    :vartype azureml.core.webservice.AksWebservice.scoring_uri: str
    :var is_default: If the Webservice is the default version for the parent AksEndpoint.
    :vartype is_default: bool
    :var traffic_percentile: What percentage of traffic to route to the Webservice in the parent AksEndpoint.
    :vartype traffic_percentile: int
    :var version_type: The version type for the Webservice in the parent AksEndpoint.
    :vartype version_type: azureml.core.webservice.aks.AksEndpoint.VersionType
    :var token_auth_enabled: Whether or not token auth is enabled for the Webservice.
    :vartype token_auth_enabled: bool
    :var environment: The Environment object that was used to create the Webservice.
    :vartype environment: azureml.core.Environment
    :var azureml.core.webservice.AksWebservice.models: A list of Models deployed to the Webservice.
    :vartype azureml.core.webservice.AksWebservice.models: builtin.list[azureml.core.Model]
    :var deployment_status: The deployment status of the Webservice.
    :vartype deployment_status: str
    :var namespace: The AKS namespace of the Webservice.
    :vartype namespace: str
    :var azureml.core.webservice.AksWebservice.swagger_uri: The swagger endpoint for the Webservice.
    :vartype azureml.core.webservice.AksWebservice.swagger_uri: str
    """

    _expected_payload_keys = Webservice._expected_payload_keys + ['appInsightsEnabled', 'authEnabled',
                                                                  'autoScaler', 'computeName',
                                                                  'containerResourceRequirements', 'dataCollection',
                                                                  'maxConcurrentRequestsPerContainer',
                                                                  'maxQueueWaitMs', 'numReplicas', 'scoringTimeoutMs',
                                                                  'scoringUri', 'livenessProbeRequirements',
                                                                  'aadAuthEnabled']
    _webservice_type = AKS_WEBSERVICE_TYPE

    def _initialize(self, workspace, obj_dict):
        """Initialize the Webservice instance.

        This is used because the constructor is used as a getter.

        :param workspace: The workspace that contains the model to deploy.
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        # Validate obj_dict with _expected_payload_keys
        AksWebservice._validate_get_payload(obj_dict)

        # Initialize common Webservice attributes
        super(AksWebservice, self)._initialize(workspace, obj_dict)

        # Initialize expected AksWebservice specific attributes
        self.enable_app_insights = obj_dict.get('appInsightsEnabled')
        self.autoscaler = AutoScaler.deserialize(obj_dict.get('autoScaler'))
        self.compute_name = obj_dict.get('computeName')
        self.container_resource_requirements = \
            ContainerResourceRequirements.deserialize(obj_dict.get('containerResourceRequirements'))
        self.liveness_probe_requirements = \
            LivenessProbeRequirements.deserialize(obj_dict.get('livenessProbeRequirements'))
        self.data_collection = DataCollection.deserialize(obj_dict.get('dataCollection'))
        self.max_concurrent_requests_per_container = obj_dict.get('maxConcurrentRequestsPerContainer')
        self.max_request_wait_time = obj_dict.get('maxQueueWaitMs')
        self.num_replicas = obj_dict.get('numReplicas')
        self.scoring_timeout_ms = obj_dict.get('scoringTimeoutMs')
        self.scoring_uri = obj_dict.get('scoringUri')
        self.is_default = obj_dict.get('isDefault')
        self.traffic_percentile = obj_dict.get('trafficPercentile')
        self.version_type = obj_dict.get('type')

        self.token_auth_enabled = obj_dict.get('aadAuthEnabled')
        env_image_request = obj_dict.get('environmentImageRequest')
        env_dict = env_image_request.get('environment') if env_image_request else None
        self.environment = Environment._deserialize_and_add_to_object(env_dict) if env_dict else None
        models = obj_dict.get('models')
        self.models = [Model.deserialize(workspace, model_payload) for model_payload in models] if models else []

        # Initialize other AKS utility attributes
        self.deployment_status = obj_dict.get('deploymentStatus')
        self.namespace = obj_dict.get('namespace')
        self.swagger_uri = '/'.join(self.scoring_uri.split('/')[:-1]) + WEBSERVICE_SWAGGER_PATH \
            if self.scoring_uri else None
        self._model_config_map = obj_dict.get('modelConfigMap')
        self._refresh_token_time = None

    def __repr__(self):
        """Return the string representation of the AksWebservice object.

        :return: String representation of the AksWebservice object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def deploy_configuration(autoscale_enabled=None, autoscale_min_replicas=None, autoscale_max_replicas=None,
                             autoscale_refresh_seconds=None, autoscale_target_utilization=None,
                             collect_model_data=None, auth_enabled=None, cpu_cores=None,
                             memory_gb=None, enable_app_insights=None, scoring_timeout_ms=None,
                             replica_max_concurrent_requests=None, max_request_wait_time=None, num_replicas=None,
                             primary_key=None, secondary_key=None, tags=None, properties=None, description=None,
                             gpu_cores=None, period_seconds=None, initial_delay_seconds=None, timeout_seconds=None,
                             success_threshold=None, failure_threshold=None, namespace=None, token_auth_enabled=None,
                             compute_target_name=None, cpu_cores_limit=None, memory_gb_limit=None,
                             blobfuse_enabled=None):
        """Create a configuration object for deploying to an AKS compute target.

        :param autoscale_enabled: Whether or not to enable autoscaling for this Webservice.
            Defaults to True if num_replicas is None.
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice.
            Defaults to 1.
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice.
            Defaults to 10.
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice.
            Defaults to 1.
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this Webservice. Defaults to 70.
        :type autoscale_target_utilization: int
        :param collect_model_data: Whether or not to enable model data collection for this Webservice.
            Defaults to False.
        :type collect_model_data: bool
        :param auth_enabled: Whether or not to enable key auth for this Webservice. Defaults to True.
        :type auth_enabled: bool
        :param cpu_cores: The number of cpu cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1.
            Corresponds to the pod core request, not the limit, in Azure Kubernetes Service.
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
            Defaults to 0.5. Corresponds to the pod memory request, not the limit, in Azure Kubernetes Service.
        :type memory_gb: float
        :param enable_app_insights: Whether or not to enable Application Insights logging for this Webservice.
            Defaults to False.
        :type enable_app_insights: bool
        :param scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice. Defaults to 60000.
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            Webservice. Defaults to 1. **Do not change this setting from the default value of 1 unless instructed by
            Microsoft Technical Support or a member of Azure Machine Learning team.**
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error. Defaults to 500.
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this Webservice. No default, if this parameter
            is not set then the autoscaler is enabled by default.
        :type num_replicas: int
        :param primary_key: A primary auth key to use for this Webservice.
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Webservice.
        :type secondary_key: str
        :param tags: Dictionary of key value tags to give this Webservice.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Webservice. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Webservice.
        :type description: str
        :param gpu_cores: The number of GPU cores to allocate for this Webservice. Defaults to 0.
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
            initiated. Defaults to 310.
        :type initial_delay_seconds: int
        :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 2 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :return: A configuration object to use when deploying a AksWebservice.
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try failureThreshold
            times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param namespace: The Kubernetes namespace in which to deploy this Webservice: up to 63 lowercase alphanumeric
            ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
        :type namespace: str
        :param token_auth_enabled: Whether or not to enable Token auth for this Webservice. If this is
            enabled, users can access this Webservice by fetching an access token using their Azure Active Directory
            credentials. Defaults to False.
        :type token_auth_enabled: bool
        :param compute_target_name: The name of the compute target to deploy to
        :type compute_target_name: str
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :param blobfuse_enabled: Whether or not to enable blobfuse for model downloading for this Webservice.
            Defaults to True
        :type blobfuse_enabled: bool
        :rtype: azureml.core.webservice.aks.AksServiceDeploymentConfiguration
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        config = AksServiceDeploymentConfiguration(autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                                                   autoscale_refresh_seconds, autoscale_target_utilization,
                                                   collect_model_data, auth_enabled, cpu_cores,
                                                   memory_gb, enable_app_insights, scoring_timeout_ms,
                                                   replica_max_concurrent_requests, max_request_wait_time,
                                                   num_replicas, primary_key, secondary_key, tags, properties,
                                                   description, gpu_cores, period_seconds, initial_delay_seconds,
                                                   timeout_seconds, success_threshold, failure_threshold, namespace,
                                                   token_auth_enabled, compute_target_name,
                                                   cpu_cores_limit, memory_gb_limit, blobfuse_enabled)
        return config

    @staticmethod
    def _deploy(workspace, name, image, deployment_config, deployment_target=None,
                overwrite=False):  # pragma: no cover
        """Deploy the Webservice.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param image:
        :type image: azureml.core.Image
        :param deployment_config:
        :type deployment_config: AksServiceDeploymentConfiguration | None
        :param deployment_target:
        :type deployment_target: azureml.core.compute.AksCompute
        :param overwrite:
        :type overwrite: bool
        :return:
        :rtype: AksWebservice
        """
        if not deployment_target and \
            not (deployment_config and hasattr(deployment_config, 'compute_target_name')
                 and deployment_config.compute_target_name is not None):
            raise WebserviceException("Must have a deployment target for an AKS web service.", logger=module_logger)
        if not deployment_config:
            deployment_config = AksWebservice.deploy_configuration()
        elif not isinstance(deployment_config, AksServiceDeploymentConfiguration):
            raise WebserviceException('Error, provided deployment configuration must be of type '
                                      'AksServiceDeploymentConfiguration in order to deploy an AKS service.',
                                      logger=module_logger)
        deployment_config.validate_image(image)

        if deployment_target:
            deployment_config.compute_target_name = deployment_target.name

        create_payload = AksWebservice._build_create_payload(name, image, deployment_config,
                                                             overwrite)
        return Webservice._deploy_webservice(workspace, name, create_payload, overwrite, AksWebservice)

    @staticmethod
    def _build_create_payload(name, image, config, overwrite):  # pragma: no cover
        """Construct the payload used to create this Webservice.

        :param name:
        :type name: str
        :param image:
        :type image: azureml.core.Image
        :param config:
        :type config: azureml.core.compute.AksServiceDeploymentConfiguration
        :return:
        :rtype: dict
        """
        from azureml._model_management._util import aks_service_payload_template
        json_payload = copy.deepcopy(aks_service_payload_template)
        json_payload['name'] = name
        json_payload['computeType'] = 'AKS'
        json_payload['computeName'] = config.compute_target_name

        json_payload['imageId'] = image.id
        properties = config.properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        json_payload['properties'] = properties

        if config.description:
            json_payload['description'] = config.description
        else:
            del (json_payload['description'])
        if config.tags:
            json_payload['kvTags'] = config.tags
        if config.num_replicas:
            json_payload['numReplicas'] = config.num_replicas
        else:
            del (json_payload['numReplicas'])
        if config.collect_model_data is None:
            del (json_payload['dataCollection'])
        else:
            json_payload['dataCollection']['storageEnabled'] = config.collect_model_data
        if config.enable_app_insights is None:
            del (json_payload['appInsightsEnabled'])
        else:
            json_payload['appInsightsEnabled'] = config.enable_app_insights
        if not config.autoscale_enabled:
            del (json_payload['autoScaler'])
        else:
            json_payload['autoScaler']['autoscaleEnabled'] = config.autoscale_enabled
            json_payload['autoScaler']['minReplicas'] = config.autoscale_min_replicas
            json_payload['autoScaler']['maxReplicas'] = config.autoscale_max_replicas
            json_payload['autoScaler']['targetUtilization'] = config.autoscale_target_utilization
            json_payload['autoScaler']['refreshPeriodInSeconds'] = config.autoscale_refresh_seconds
            if 'numReplicas' in json_payload:
                del (json_payload['numReplicas'])
        if config.auth_enabled is None:
            del (json_payload['authEnabled'])
        else:
            json_payload['authEnabled'] = config.auth_enabled
        if config.token_auth_enabled is None:
            del json_payload['aadAuthEnabled']
        else:
            json_payload['aadAuthEnabled'] = config.token_auth_enabled
        if config.cpu_cores or config.cpu_cores_limit or \
                config.memory_gb or config.memory_gb_limit or \
                config.gpu_cores:
            if config.cpu_cores:
                json_payload['containerResourceRequirements']['cpu'] = config.cpu_cores
            else:
                del (json_payload['containerResourceRequirements']['cpu'])
            if config.cpu_cores_limit:
                json_payload['containerResourceRequirements']['cpuLimit'] = config.cpu_cores_limit
            else:
                del (json_payload['containerResourceRequirements']['cpuLimit'])
            if config.memory_gb:
                json_payload['containerResourceRequirements']['memoryInGB'] = config.memory_gb
            else:
                del (json_payload['containerResourceRequirements']['memoryInGB'])
            if config.memory_gb_limit:
                json_payload['containerResourceRequirements']['memoryInGBLimit'] = config.memory_gb_limit
            else:
                del (json_payload['containerResourceRequirements']['memoryInGBLimit'])
            if config.gpu_cores:
                json_payload['containerResourceRequirements']['gpu'] = config.gpu_cores
            else:
                del (json_payload['containerResourceRequirements']['gpu'])
        else:
            del (json_payload['containerResourceRequirements'])

        if config.period_seconds or config.initial_delay_seconds or config.timeout_seconds \
                or config.failure_threshold or config.success_threshold:
            if config.period_seconds:
                json_payload['livenessProbeRequirements']['periodSeconds'] = config.period_seconds
            else:
                del (json_payload['livenessProbeRequirements']['periodSeconds'])
            if config.initial_delay_seconds:
                json_payload['livenessProbeRequirements']['initialDelaySeconds'] = config.initial_delay_seconds
            else:
                del (json_payload['livenessProbeRequirements']['initialDelaySeconds'])
            if config.timeout_seconds:
                json_payload['livenessProbeRequirements']['timeoutSeconds'] = config.timeout_seconds
            else:
                del (json_payload['livenessProbeRequirements']['timeoutSeconds'])
            if config.failure_threshold:
                json_payload['livenessProbeRequirements']['failureThreshold'] = config.failure_threshold
            else:
                del (json_payload['livenessProbeRequirements']['failureThreshold'])
            if config.success_threshold:
                json_payload['livenessProbeRequirements']['successThreshold'] = config.success_threshold
            else:
                del (json_payload['livenessProbeRequirements']['successThreshold'])
        else:
            del (json_payload['livenessProbeRequirements'])

        json_payload['maxConcurrentRequestsPerContainer'] = config.replica_max_concurrent_requests
        if config.max_request_wait_time:
            json_payload['maxQueueWaitMs'] = config.max_request_wait_time
        else:
            del (json_payload['maxQueueWaitMs'])
        if config.namespace:
            json_payload['namespace'] = config.namespace
        else:
            del (json_payload['namespace'])
        if config.primary_key:
            json_payload['keys']['primaryKey'] = config.primary_key
            json_payload['keys']['secondaryKey'] = config.secondary_key
        else:
            del (json_payload['keys'])
        if config.scoring_timeout_ms:
            json_payload['scoringTimeoutMs'] = config.scoring_timeout_ms
        else:
            del (json_payload['scoringTimeoutMs'])

        if overwrite:
            json_payload['overwrite'] = overwrite
        else:
            del (json_payload['overwrite'])
        return json_payload

    def run(self, input_data):
        """Call this Webservice with the provided input.

        :param input_data: The input to call the Webservice with
        :type input_data: varies
        :return: The result of calling the Webservice
        :rtype: dict
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if not self.scoring_uri:
            raise WebserviceException('Error attempting to call webservice, scoring_uri unavailable. '
                                      'This could be due to a failed deployment, or the service is not ready yet.\n'
                                      'Current State: {}\n'
                                      'Errors: {}'.format(self.state, self.error), logger=module_logger)

        resp = ClientBase._execute_func(self._webservice_session.post, self.scoring_uri, data=input_data)

        if resp.status_code == 401:
            if self.auth_enabled:
                service_keys = self.get_keys()
                self._session.headers.update({'Authorization': 'Bearer ' + service_keys[0]})
            elif self.token_auth_enabled:
                service_token, refresh_token_time = self.get_token()
                self._refresh_token_time = refresh_token_time
                self._session.headers.update({'Authorization': 'Bearer ' + service_token})
            resp = ClientBase._execute_func(self._webservice_session.post, self.scoring_uri, data=input_data)

        if resp.status_code == 200:
            return resp.json()
        else:
            raise WebserviceException('Received bad response from service. More information can be found by calling '
                                      '`.get_logs()` on the webservice object.\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    def update(self, image=None, autoscale_enabled=None, autoscale_min_replicas=None, autoscale_max_replicas=None,
               autoscale_refresh_seconds=None, autoscale_target_utilization=None, collect_model_data=None,
               auth_enabled=None, cpu_cores=None, memory_gb=None,
               enable_app_insights=None, scoring_timeout_ms=None, replica_max_concurrent_requests=None,
               max_request_wait_time=None, num_replicas=None, tags=None,
               properties=None, description=None, models=None, inference_config=None, gpu_cores=None,
               period_seconds=None, initial_delay_seconds=None, timeout_seconds=None, success_threshold=None,
               failure_threshold=None, namespace=None, token_auth_enabled=None, cpu_cores_limit=None,
               memory_gb_limit=None, **kwargs):
        """Update the Webservice with provided properties.

        Values left as None will remain unchanged in this Webservice.

        :param image: A new Image to deploy to the Webservice
        :type image: azureml.core.Image
        :param autoscale_enabled: Enable or disable autoscaling of this Webservice
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this Webservice
        :type autoscale_target_utilization: int
        :param collect_model_data: Enable or disable model data collection for this Webservice
        :type collect_model_data: bool
        :param auth_enabled: Whether or not to enable auth for this Webservice
        :type auth_enabled: bool
        :param cpu_cores: The number of cpu cores to allocate for this Webservice. Can be a decimal
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal
        :type memory_gb: float
        :param enable_app_insights: Whether or not to enable Application Insights logging for this Webservice
        :type enable_app_insights: bool
        :param scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            Webservice.
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this Webservice
        :type num_replicas: int
        :param tags: Dictionary of key value tags to give this Webservice. Will replace existing tags.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to add to existing properties dictionary
        :type properties: dict[str, str]
        :param description: A description to give this Webservice
        :type description: str
        :param models: A list of Model objects to package with the updated service
        :type models: builtin.list[azureml.core.Model]
        :param inference_config: An InferenceConfig object used to provide the required model deployment properties.
        :type inference_config: azureml.core.model.InferenceConfig
        :param gpu_cores: The number of gpu cores to allocate for this Webservice
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: Number of seconds after the container has started before liveness probes are
            initiated.
        :type initial_delay_seconds: int
        :param timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 1 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try failureThreshold
            times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param namespace: The Kubernetes namespace in which to deploy this Webservice: up to 63 lowercase alphanumeric
            ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
        :type namespace: str
        :param token_auth_enabled: Whether or not to enable Token auth for this Webservice. If this is
            enabled, users can access this Webservice by fetching access token using their Azure Active Directory
            credentials. Defaults to False
        :type token_auth_enabled: bool
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :param kwargs: include params to support migrating AKS web service to Kubernetes online endpoint and
            deployment. is_migration=True|False, compute_target=<compute target with AzureML extension installed to
            host migrated Kubernetes online endpoint and deployment>.
        :type kwargs: varies
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        is_migration = kwargs.get('is_migration', False)
        compute_target = kwargs.get('compute_target', None)

        if is_migration:
            if compute_target is None:
                raise WebserviceException('Must have a compute target to migrate AKS web service.',
                                          logger=module_logger)
            self._migrate_service(compute_target)
            return

        if not image and autoscale_enabled is None and not autoscale_min_replicas and not autoscale_max_replicas \
                and not autoscale_refresh_seconds and not autoscale_target_utilization and collect_model_data is None \
                and auth_enabled is None and not cpu_cores and not memory_gb and not gpu_cores \
                and enable_app_insights is None and not scoring_timeout_ms and not replica_max_concurrent_requests \
                and not max_request_wait_time and not num_replicas and tags is None and properties is None \
                and not description and not period_seconds and not initial_delay_seconds and not timeout_seconds \
                and models is None and inference_config is None and not failure_threshold and not success_threshold \
                and not namespace and token_auth_enabled is None and cpu_cores_limit is None \
                and memory_gb_limit is None:
            raise WebserviceException('No parameters provided to update.', logger=module_logger)

        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        self._validate_update(image, autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                              autoscale_refresh_seconds, autoscale_target_utilization, collect_model_data, cpu_cores,
                              memory_gb, enable_app_insights, scoring_timeout_ms, replica_max_concurrent_requests,
                              max_request_wait_time, num_replicas, tags, properties, description, models,
                              inference_config, gpu_cores, period_seconds, initial_delay_seconds, timeout_seconds,
                              success_threshold, failure_threshold, namespace, auth_enabled, token_auth_enabled,
                              cpu_cores_limit, memory_gb_limit)

        patch_list = []

        if inference_config:
            if not models:
                models = self.image.models if self.image else self.models

            if self.environment:  # Use the new environment handling
                inference_config, _ = convert_parts_to_environment(self.name, inference_config)

                environment_image_request = \
                    inference_config._build_environment_image_request(self.workspace, [model.id for model in models])
                patch_list.append({'op': 'replace', 'path': '/environmentImageRequest',
                                   'value': environment_image_request})
            else:  # pragma: no cover
                # Use the old image handling
                image = Image.create(self.workspace, self.name, models, inference_config)
                image.wait_for_creation(True)
                if image.creation_state != 'Succeeded':
                    raise WebserviceException('Error occurred creating model package {} for service update. More '
                                              'information can be found here: {}, generated DockerFile can be '
                                              'found here: {}'.format(image.id,
                                                                      image.image_build_log_uri,
                                                                      image.generated_dockerfile_uri),
                                              logger=module_logger)
        elif models is not None:
            # No-code deploy.
            environment_image_request = build_and_validate_no_code_environment_image_request(models)
            patch_list.append({'op': 'replace', 'path': '/environmentImageRequest',
                               'value': environment_image_request})

        properties = properties or {}
        properties.update(global_tracking_info_registry.gather_all())

        if image:
            patch_list.append({'op': 'replace', 'path': '/imageId', 'value': image.id})
        if autoscale_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/autoScaler/autoscaleEnabled', 'value': autoscale_enabled})
        if autoscale_min_replicas:
            patch_list.append({'op': 'replace', 'path': '/autoScaler/minReplicas', 'value': autoscale_min_replicas})
        if autoscale_max_replicas:
            patch_list.append({'op': 'replace', 'path': '/autoScaler/maxReplicas', 'value': autoscale_max_replicas})
        if autoscale_refresh_seconds:
            patch_list.append({'op': 'replace', 'path': '/autoScaler/refreshPeriodInSeconds',
                               'value': autoscale_refresh_seconds})
        if autoscale_target_utilization:
            patch_list.append({'op': 'replace', 'path': '/autoScaler/targetUtilization',
                               'value': autoscale_target_utilization})
        if collect_model_data is not None:
            patch_list.append({'op': 'replace', 'path': '/dataCollection/storageEnabled', 'value': collect_model_data})

        if auth_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/authEnabled', 'value': auth_enabled})
        if token_auth_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/aadAuthEnabled', 'value': token_auth_enabled})

        if cpu_cores:
            patch_list.append({'op': 'replace', 'path': '/containerResourceRequirements/cpu', 'value': cpu_cores})
        if cpu_cores_limit:
            patch_list.append({'op': 'replace', 'path': '/containerResourceRequirements/cpuLimit',
                               'value': cpu_cores_limit})
        if memory_gb:
            patch_list.append({'op': 'replace', 'path': '/containerResourceRequirements/memoryInGB',
                               'value': memory_gb})
        if memory_gb_limit:
            patch_list.append({'op': 'replace', 'path': '/containerResourceRequirements/memoryInGBLimit',
                               'value': memory_gb_limit})

        if gpu_cores:
            patch_list.append({'op': 'replace', 'path': '/containerResourceRequirements/gpu', 'value': gpu_cores})
        if enable_app_insights is not None:
            patch_list.append({'op': 'replace', 'path': '/appInsightsEnabled', 'value': enable_app_insights})
        if scoring_timeout_ms:
            patch_list.append({'op': 'replace', 'path': '/scoringTimeoutMs', 'value': scoring_timeout_ms})
        if replica_max_concurrent_requests:
            patch_list.append({'op': 'replace', 'path': '/maxConcurrentRequestsPerContainer',
                               'value': replica_max_concurrent_requests})
        if max_request_wait_time:
            patch_list.append({'op': 'replace', 'path': '/maxQueueWaitMs',
                               'value': max_request_wait_time})
        if num_replicas:
            patch_list.append({'op': 'replace', 'path': '/numReplicas', 'value': num_replicas})
        if period_seconds:
            patch_list.append({'op': 'replace', 'path': '/livenessProbeRequirements/periodSeconds',
                               'value': period_seconds})
        if initial_delay_seconds:
            patch_list.append({'op': 'replace', 'path': '/livenessProbeRequirements/initialDelaySeconds',
                               'value': initial_delay_seconds})
        if timeout_seconds:
            patch_list.append({'op': 'replace', 'path': '/livenessProbeRequirements/timeoutSeconds',
                               'value': timeout_seconds})
        if success_threshold:
            patch_list.append({'op': 'replace', 'path': '/livenessProbeRequirements/successThreshold',
                               'value': success_threshold})
        if failure_threshold:
            patch_list.append({'op': 'replace', 'path': '/livenessProbeRequirements/failureThreshold',
                               'value': failure_threshold})
        if namespace:
            patch_list.append({'op': 'replace', 'path': '/namespace', 'value': namespace})
        if tags is not None:
            patch_list.append({'op': 'replace', 'path': '/kvTags', 'value': tags})
        if properties is not None:
            for key in properties:
                patch_list.append({'op': 'add', 'path': '/properties/{}'.format(key), 'value': properties[key]})
        if description:
            patch_list.append({'op': 'replace', 'path': '/description', 'value': description})

        Webservice._check_for_webservice(self.workspace, self.name, self.compute_type,
                                         patch_list, SERVICE_REQUEST_OPERATION_UPDATE)
        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list)

        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            service_payload = json.loads(content)
            if 'operationId' in service_payload:
                create_operation_status_id = service_payload['operationId']
                base_url = '/'.join(self._mms_endpoint.split('/')[:-2])
                operation_url = base_url + '/operations/{}'.format(create_operation_status_id)
                self._operation_endpoint = operation_url
            self.update_deployment_state()
        elif resp.status_code == 202:
            if 'Operation-Location' in resp.headers:
                operation_location = resp.headers['Operation-Location']
            else:
                raise WebserviceException('Missing response header key: Operation-Location', logger=module_logger)
            create_operation_status_id = operation_location.split('/')[-1]
            base_url = '/'.join(self._mms_endpoint.split('/')[:-2])
            operation_url = base_url + '/operations/{}'.format(create_operation_status_id)
            self._operation_endpoint = operation_url
            self.update_deployment_state()
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    def _migrate_service(self, compute_target):
        """
        Migrate this Webservice to Kubernetes online endpoint and deployment.

        A :class:`azureml.exceptions.WebserviceException` is raised if there is a problem migrating the
        Webservice.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        headers = self._auth.get_authentication_header()
        params = {}
        base_url = _get_mms_url(self.workspace)
        migration_endpoint = base_url + '/services/migrate'

        resp = ClientBase._execute_func(get_requests_session().post, migration_endpoint, headers=headers,
                                        params=params, timeout=MMS_MIGRATION_TIMEOUT_SECONDS,
                                        json=AksWebservice._generate_migration_payload(self.name,
                                                                                       compute_target,
                                                                                       self.workspace))

        if resp.status_code == 200:
            self.state = 'Migrated'
        elif resp.status_code == 202:
            self.state = 'Migrating'
            if 'Operation-Location' in resp.headers:
                operation_location = resp.headers['Operation-Location']
            else:
                raise WebserviceException('Missing response header key: Operation-Location', resp.status_code,
                                          logger=module_logger)
            create_operation_status_id = operation_location.split('/')[-1]
            operation_url = base_url + '/operations/{}'.format(create_operation_status_id)

            self._operation_endpoint = operation_url
            print('Request submitted, please run wait_for_deployment(show_output=True) to get migration status.')

        elif resp.status_code == 404:
            print('No service with name {} found to migrate.'.format(self.name))
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    def _validate_update(self, image, autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                         autoscale_refresh_seconds, autoscale_target_utilization, collect_model_data, cpu_cores,
                         memory_gb, enable_app_insights, scoring_timeout_ms, replica_max_concurrent_requests,
                         max_request_wait_time, num_replicas, tags, properties, description, models,
                         inference_config, gpu_cores, period_seconds, initial_delay_seconds, timeout_seconds,
                         success_threshold, failure_threshold, namespace, auth_enabled, token_auth_enabled,
                         cpu_cores_limit, memory_gb_limit):
        """Validate the values provided to update the webservice.

        :param image:
        :type image: azureml.core.Image
        :param autoscale_enabled:
        :type autoscale_enabled: bool
        :param autoscale_min_replicas:
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas:
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds:
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization:
        :type autoscale_target_utilization: int
        :param collect_model_data:
        :type collect_model_data: bool
        :param cpu_cores:
        :type cpu_cores: float
        :param memory_gb:
        :type memory_gb: float
        :param enable_app_insights:
        :type enable_app_insights: bool
        :param scoring_timeout_ms:
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests:
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error
        :type max_request_wait_time: int
        :param num_replicas:
        :type num_replicas: int
        :param tags:
        :type tags: dict[str, str]
        :param properties:
        :type properties: dict[str, str]
        :param description:
        :type description: str
        :param models: A list of Model objects to package with this image. Can be an empty list
        :type models: :class:`list[azureml.core.Model]`
        :param inference_config: An InferenceConfig object used to determine required model properties.
        :type inference_config: azureml.core.model.InferenceConfig
        :param: gpu_cores
        :type: int
        :param period_seconds:
        :type period_seconds: int
        :param initial_delay_seconds:
        :type initial_delay_seconds: int
        :param timeout_seconds:
        :type timeout_seconds: int
        :param success_threshold:
        :type success_threshold: int
        :param failure_threshold:
        :type failure_threshold: int
        :param namespace:
        :type namespace: str
        :param auth_enabled: If Key auth is enabled.
        :type auth_enabled: bool
        :param token_auth_enabled: Whether or not to enable Token auth for this Webservice. If this is
            enabled, users can access this Webservice by fetching access token using their Azure Active Directory
            credentials. Defaults to False
        :type token_auth_enabled: bool
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        """
        error = ""
        if image and self.environment:
            error += 'Error, unable to use Image object to update Webservice created with Environment object.\n'
        if inference_config and inference_config.environment and self.image:
            error += 'Error, unable to use Environment object to update Webservice created with Image object.\n'
        if image and inference_config:
            error += 'Error, unable to pass both an Image object and an InferenceConfig object to update.\n'
        if cpu_cores is not None:
            if cpu_cores <= 0:
                error += 'Error, cpu_cores must be greater than zero.\n'
            if self.container_resource_requirements is not None and \
                    self.container_resource_requirements.cpu_limit is not None \
                    and cpu_cores > self.container_resource_requirements.cpu_limit:
                error += 'Error, cpu_cores must be ' \
                         'less than or equal to container_resource_requirements.cpu_limit.\n'
        if cpu_cores_limit is not None:
            if cpu_cores_limit <= 0:
                error += 'Error, cpu_cores_limit must be greater than zero.\n'
            if cpu_cores is not None and cpu_cores_limit < cpu_cores:
                error += 'Error, cpu_cores_limit must be greater than or equal to cpu_cores.\n'
            if self.container_resource_requirements is not None and \
                    self.container_resource_requirements.cpu is not None \
                    and cpu_cores_limit < self.container_resource_requirements.cpu:
                error += 'Error, cpu_cores_limit must be ' \
                         'greater than or equal to container_resource_requirements.cpu.\n'
        if memory_gb is not None:
            if memory_gb <= 0:
                error += 'Error, memory_gb must be greater than zero.\n'
            if self.container_resource_requirements and self.container_resource_requirements.memory_in_gb_limit \
                    and memory_gb > self.container_resource_requirements.memory_in_gb_limit:
                error += 'Error, memory_gb must be ' \
                         'less than or equal to container_resource_requirements.memory_in_gb_limit.\n'
        if memory_gb_limit is not None:
            if memory_gb_limit <= 0:
                error += 'Error, memory_gb_limit must be greater than zero.\n'
            elif memory_gb and memory_gb_limit < memory_gb:
                error += 'Error, memory_gb_limit must be greater than or equal to memory_gb.\n'
            elif self.container_resource_requirements and self.container_resource_requirements.memory_in_gb \
                    and memory_gb_limit < self.container_resource_requirements.memory_in_gb:
                error += 'Error, memory_gb_limit must be ' \
                         'greater than or equal to container_resource_requirements.memory_in_gb.\n'
        if gpu_cores is not None and gpu_cores < 0:
            error += 'Error, gpu_cores must be greater than or equal to zero.\n'
        if scoring_timeout_ms is not None and scoring_timeout_ms <= 0:
            error += 'Error, scoring_timeout_ms must be greater than zero.\n'
        if replica_max_concurrent_requests is not None and replica_max_concurrent_requests <= 0:
            error += 'Error, replica_max_concurrent_requests must be greater than zero.\n'
        if max_request_wait_time is not None and max_request_wait_time <= 0:
            error += 'Error, max_request_wait_time must be greater than zero.\n'
        if num_replicas is not None and num_replicas <= 0:
            error += 'Error, num_replicas must be greater than zero.\n'
        if period_seconds is not None and period_seconds <= 0:
            error += 'Error, period_seconds must be greater than zero.\n'
        if timeout_seconds is not None and timeout_seconds <= 0:
            error += 'Error, timeout_seconds must be greater than zero.\n'
        if initial_delay_seconds is not None and initial_delay_seconds <= 0:
            error += 'Error, initial_delay_seconds must be greater than zero.\n'
        if success_threshold is not None and success_threshold <= 0:
            error += 'Error, success_threshold must be greater than zero.\n'
        if failure_threshold is not None and failure_threshold <= 0:
            error += 'Error, failure_threshold must be greater than zero.\n'
        if namespace and not re.match(NAMESPACE_REGEX, namespace):
            error += 'Error, namespace must be a valid Kubernetes namespace. ' \
                     'Regex for validation is ' + NAMESPACE_REGEX + '\n'
        if autoscale_enabled:
            if num_replicas:
                error += 'Error, autoscale enabled and num_replicas provided.\n'
            if autoscale_min_replicas is not None and autoscale_min_replicas <= 0:
                error += 'Error, autoscale_min_replicas must be greater than zero.\n'
            if autoscale_max_replicas is not None and autoscale_max_replicas <= 0:
                error += 'Error, autoscale_max_replicas must be greater than zero.\n'
            if autoscale_min_replicas and autoscale_max_replicas and \
                    autoscale_min_replicas > autoscale_max_replicas:
                error += 'Error, autoscale_min_replicas cannot be greater than autoscale_max_replicas.\n'
            if autoscale_refresh_seconds is not None and autoscale_refresh_seconds <= 0:
                error += 'Error, autoscale_refresh_seconds must be greater than zero.\n'
            if autoscale_target_utilization is not None and autoscale_target_utilization <= 0:
                error += 'Error, autoscale_target_utilization must be greater than zero.\n'
        else:
            if autoscale_enabled is False and not num_replicas:
                error += 'Error, autoscale disabled but num_replicas not provided.\n'
            if autoscale_min_replicas:
                error += 'Error, autoscale_min_replicas provided without enabling autoscaling.\n'
            if autoscale_max_replicas:
                error += 'Error, autoscale_max_replicas provided without enabling autoscaling.\n'
            if autoscale_refresh_seconds:
                error += 'Error, autoscale_refresh_seconds provided without enabling autoscaling.\n'
            if autoscale_target_utilization:
                error += 'Error, autoscale_target_utilization provided without enabling autoscaling.\n'
        if inference_config is None and models and \
                (len(models) != 1 or models[0].model_framework not in Model._SUPPORTED_FRAMEWORKS_FOR_NO_CODE_DEPLOY):
            error += 'Error, both "models" and "inference_config" inputs must be provided in order ' \
                     'to update the models.\n'
        if token_auth_enabled and auth_enabled:
            error += 'Error, cannot set both token_auth_enabled and auth_enabled.\n'
        elif token_auth_enabled and (self.auth_enabled and auth_enabled is not False):
            error += 'Error, cannot set token_auth_enabled without disabling key auth (set auth_enabled to False).\n'
        elif auth_enabled and (self.token_auth_enabled and token_auth_enabled is not False):
            error += 'Error, cannot set token_auth_enabled without disabling key auth (set auth_enabled to False).\n'

        if error:
            raise WebserviceException(error, logger=module_logger)

    def add_tags(self, tags):
        """Add key value pairs to this Webservice's tags dictionary.

        Raises a :class:`azureml.exceptions.WebserviceException`.

        :param tags: The dictionary of tags to add.
        :type tags: dict[str, str]
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        updated_tags = self._add_tags(tags)
        self.tags = updated_tags
        self.update(tags=updated_tags)

        print('Webservice tag add operation complete.')

    def remove_tags(self, tags):
        """Remove the specified keys from this Webservice's dictionary of tags.

        :param tags: The list of keys to remove
        :type tags: builtin.list[str]
        """
        updated_tags = self._remove_tags(tags)
        self.tags = updated_tags
        self.update(tags=updated_tags)

        print('Webservice tag remove operation complete.')

    def add_properties(self, properties):
        """Add key value pairs to this Webservice's properties dictionary.

        :param properties: The dictionary of properties to add.
        :type properties: dict[str, str]
        """
        updated_properties = self._add_properties(properties)
        self.properties = updated_properties
        self.update(properties=updated_properties)

        print('Webservice property add operation complete.')

    def serialize(self):
        """Convert this Webservice into a JSON serialized dictionary.

        :return: The JSON representation of this Webservice.
        :rtype: dict
        """
        properties = super(AksWebservice, self).serialize()
        autoscaler = self.autoscaler.serialize() if self.autoscaler else None
        container_resource_requirements = self.container_resource_requirements.serialize() \
            if self.container_resource_requirements else None
        liveness_probe_requirements = self.liveness_probe_requirements.serialize() \
            if self.liveness_probe_requirements else None
        data_collection = self.data_collection.serialize() if self.data_collection else None
        env_details = Environment._serialize_to_dict(self.environment) if self.environment else None
        model_details = [model.serialize() for model in self.models] if self.models else None
        aks_properties = {'appInsightsEnabled': self.enable_app_insights, 'authEnabled': self.auth_enabled,
                          'autoScaler': autoscaler, 'computeName': self.compute_name,
                          'containerResourceRequirements': container_resource_requirements,
                          'dataCollection': data_collection, 'imageId': self.image_id,
                          'maxConcurrentRequestsPerContainer': self.max_concurrent_requests_per_container,
                          'maxQueueWaitMs': self.max_request_wait_time,
                          'livenessProbeRequirements': liveness_probe_requirements,
                          'numReplicas': self.num_replicas, 'deploymentStatus': self.deployment_status,
                          'scoringTimeoutMs': self.scoring_timeout_ms, 'scoringUri': self.scoring_uri,
                          'aadAuthEnabled': self.token_auth_enabled, 'environmentDetails': env_details,
                          'modelDetails': model_details, 'isDefault': self.is_default,
                          'trafficPercentile': self.traffic_percentile, 'versionType': self.version_type}
        properties.update(aks_properties)
        return properties

    def get_token(self):  # pragma: no cover
        """DEPRECATED. Use ``get_access_token`` method instead.

        Retrieve auth token for this Webservice.

        :return: The auth token for this Webservice and when to refresh it.
        :rtype: str, datetime
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        warnings.warn("get_token() method has been deprecated and will be removed in a future release. "
                      + "Please migrate to get_access_token() method.", category=DeprecationWarning)
        service_token = self._internal_get_access_token()
        return service_token.access_token, service_token.refresh_after

    def get_access_token(self):
        """Retrieve auth token for this Webservice.

        :return: An object describing the auth token for this Webservice.
        :rtype: azureml.core.webservice.aks.AksServiceAccessToken
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        return self._internal_get_access_token()

    def _internal_get_access_token(self):
        """Retrieve auth token for this Webservice.

        :return: An object describing the auth token for this Webservice.
        :class:`azureml.core.webservice.aks.AksServiceAccessToken`
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        headers = self._auth.get_authentication_header()
        params = {}
        token_url = self._mms_endpoint + '/token'

        try:
            resp = ClientBase._execute_func(get_requests_session().post, token_url,
                                            params=params, headers=headers)

            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            auth_token_result = json.loads(content)
            return AksServiceAccessToken.deserialize(auth_token_result)
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    @staticmethod
    def _generate_migration_payload(service_name, compute_target, workspace):
        compute_target_uri = '/subscriptions/{}/resourceGroups/{}/providers/' \
            'Microsoft.MachineLearningServices/workspaces/{}/computes/{}'.format(workspace.subscription_id,
                                                                                 workspace.resource_group,
                                                                                 workspace.name,
                                                                                 compute_target)
        payload = {'serviceName': service_name,
                   'computeTarget': compute_target_uri}
        return payload


class AksEndpoint(AksWebservice):
    """Represents a collection of web service versions behind the same endpoint running on Azure Kubernetes Service.

    Whereas a :class:`azureml.core.webservice.aks.AksWebservice` deploys a single service with a single
    scoring endpoint, the AksEndpoint class enables you to deploy multiple web service versions behind the
    same scoring endpoint. Each web service version can be configured to serve a percentage of the traffic
    so you can deploy models in a controlled fashion, for example, for A/B testing. The AksEndpoint allows deployment
    from a model object similar to AksWebservice.

    :var versions: A dictionary of version name to version object. Contains all of the versions deployed as a part of
        this Endpoint.
    :vartype versions: dict[str, azureml.core.webservice.aks.AksWebservice]
    """

    class VersionType(Enum):
        """Defines the version type of an AksEndpoint.

        When creating or updating a version of an :class:`azureml.core.webservice.aks.AksWebservice`, you
        can specify whether the version is a control version or not.
        """

        control = "Control"
        treatment = "Treatment"

    _expected_payload_keys = Webservice._expected_payload_keys + ['appInsightsEnabled', 'authEnabled', 'computeName',
                                                                  'scoringUri', 'aadAuthEnabled']
    _webservice_type = AKS_ENDPOINT_TYPE

    def _initialize(self, workspace, obj_dict):
        """Initialize the Endpoint instance.

        This is used because the constructor is used as a getter.

        :param workspace: The workspace with the model to deploy.
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        # Validate obj_dict with _expected_payload_keys
        AksEndpoint._validate_get_payload(obj_dict)

        # Initialize common Webservice attributes
        self.auth_enabled = obj_dict.get('authEnabled')
        self.compute_type = obj_dict.get('computeType')
        self.created_time = parse(obj_dict.get('createdTime'))
        self.description = obj_dict.get('description')
        self.tags = obj_dict.get('kvTags')
        self.name = obj_dict.get('name')
        self.properties = obj_dict.get('properties')

        # Common amongst Webservice classes but optional payload keys
        self.error = obj_dict.get('error')
        self.state = obj_dict.get('state')
        self.updated_time = parse(obj_dict['updatedTime']) if 'updatedTime' in obj_dict else None

        # Utility payload keys
        self._auth = workspace._auth_object
        self._mms_endpoint = _get_mms_url(workspace) + '/services/{}'.format(self.name)
        self._operation_endpoint = None
        self.workspace = workspace
        self._session = None
        self.image = None

        # Initialize expected AksEndpoint specific attributes
        self.enable_app_insights = obj_dict.get('appInsightsEnabled')
        self.compute_name = obj_dict.get('computeName')
        self.scoring_uri = obj_dict.get('scoringUri')
        self.token_auth_enabled = obj_dict.get('aadAuthEnabled')
        version_payloads = obj_dict.get('versions', {})
        self.versions = {}

        if version_payloads is not None:
            for version_name in version_payloads:
                version = super(Webservice, self).__new__(AksWebservice)
                version._initialize(workspace, version_payloads[version_name])
                self.versions[version_name] = version

        # Initialize other AKS utility attributes
        self.deployment_status = obj_dict.get('deploymentStatus')
        self.namespace = obj_dict.get('namespace')
        self.swagger_uri = '/'.join(self.scoring_uri.split('/')[:-1]) + WEBSERVICE_SWAGGER_PATH \
            if self.scoring_uri else None
        self._refresh_token_time = None

    @staticmethod
    def deploy_configuration(autoscale_enabled=None, autoscale_min_replicas=None, autoscale_max_replicas=None,
                             autoscale_refresh_seconds=None, autoscale_target_utilization=None,
                             collect_model_data=None, auth_enabled=None, cpu_cores=None,
                             memory_gb=None, enable_app_insights=None, scoring_timeout_ms=None,
                             replica_max_concurrent_requests=None, max_request_wait_time=None, num_replicas=None,
                             primary_key=None, secondary_key=None, tags=None, properties=None, description=None,
                             gpu_cores=None, period_seconds=None, initial_delay_seconds=None, timeout_seconds=None,
                             success_threshold=None, failure_threshold=None, namespace=None, token_auth_enabled=None,
                             version_name=None, traffic_percentile=None, compute_target_name=None,
                             cpu_cores_limit=None, memory_gb_limit=None):
        """Create a configuration object for deploying to an AKS compute target.

        :param autoscale_enabled: Whether or not to enable autoscaling for this version in an Endpoint.
            Defaults to True if ``num_replicas`` is None.
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this version in an
            Endpoint. Defaults to 1.
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this version in an
            Endpoint. Defaults to 10.
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this version in an Endpoint.
            Defaults to 1.
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this version in an Endpoint. Defaults to 70.
        :type autoscale_target_utilization: int
        :param collect_model_data: Whether or not to enable model data collection for this version in an Endpoint.
            Defaults to False.
        :type collect_model_data: bool
        :param auth_enabled: Whether or not to enable key auth for this version in an Endpoint. Defaults to True.
        :type auth_enabled: bool
        :param cpu_cores: The number of cpu cores to allocate for this version in an Endpoint. Can be a decimal.
            Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this version in an Endpoint. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param enable_app_insights: Whether or not to enable ApplicationInsights logging for this version in an
            Endpoint. Defaults to False.
        :type enable_app_insights: bool
        :param scoring_timeout_ms: A timeout to enforce scoring calls to this version in an Endpoint. Defaults to 60000
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            version in an Endpoint. Defaults to 1. **Do not change this setting from the default value of 1 unless
            instructed by Microsoft Technical Support or a member of Azure Machine Learning team.**
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error. Defaults to 500.
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this version in an Endpoint. No default, if this
            parameter is not set then the autoscaler is enabled by default.
        :type num_replicas: int
        :param primary_key: A primary auth key to use for this Endpoint.
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Endpoint.
        :type secondary_key: str
        :param tags: Dictionary of key value tags to give this Endpoint.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Endpoint. These properties cannot
            be changed after deployment, however new key value pairs can be added
        :type properties: dict[str, str]
        :param description: A description to give this Endpoint.
        :type description: str
        :param gpu_cores: The number of GPU cores to allocate for this version in an Endpoint. Defaults to 0.
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: Number of seconds after the container has started before liveness probes are
            initiated. Defaults to 310.
        :type initial_delay_seconds: int
        :param timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 2 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
            ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param namespace: The Kubernetes namespace in which to deploy this Endpoint: up to 63 lowercase alphanumeric
            ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
        :type namespace: str
        :param token_auth_enabled: Whether or not to enable Token auth for this Endpoint. If this is
            enabled, users can access this Endpoint by fetching access token using their Azure Active Directory
            credentials. Defaults to False.
        :type token_auth_enabled: bool
        :param version_name: The name of the version in an endpoint.
        :type version_name: str
        :param traffic_percentile: the amount of traffic the version takes in an endpoint.
        :type traffic_percentile: float
        :param compute_target_name: The name of the compute target to deploy to
        :type compute_target_name: str
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :rtype: azureml.core.webservice.aks.AksEndpointDeploymentConfiguration
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        config = AksEndpointDeploymentConfiguration(autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                                                    autoscale_refresh_seconds, autoscale_target_utilization,
                                                    collect_model_data, auth_enabled, cpu_cores, memory_gb,
                                                    enable_app_insights, scoring_timeout_ms,
                                                    replica_max_concurrent_requests, max_request_wait_time,
                                                    num_replicas, primary_key, secondary_key, tags, properties,
                                                    description, gpu_cores, period_seconds, initial_delay_seconds,
                                                    timeout_seconds, success_threshold, failure_threshold, namespace,
                                                    token_auth_enabled, version_name, traffic_percentile,
                                                    compute_target_name, cpu_cores_limit, memory_gb_limit)
        return config

    def update(self, auth_enabled=None, token_auth_enabled=None, enable_app_insights=None, description=None, tags=None,
               properties=None):
        """Update the Endpoint with provided properties.

        Values left as None will remain unchanged in this Endpoint

        :param auth_enabled: Whether or not to enable key auth for this version in an Endpoint. Defaults to True.
        :type auth_enabled: bool
        :param token_auth_enabled: Whether or not to enable Token auth for this Endpoint. If this is
            enabled, users can access this Endpoint by fetching access token using their Azure Active Directory
            credentials. Defaults to False.
        :type token_auth_enabled: bool
        :param enable_app_insights: Whether or not to enable Application Insights logging for this version in an
            Endpoint. Defaults to False.
        :type enable_app_insights: bool
        :param description: A description to give this Endpoint.
        :type description: str
        :param tags: Dictionary of key value tags to give this Endpoint.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Endpoint. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if auth_enabled is None and enable_app_insights is None and tags is None and properties is None \
                and not description and token_auth_enabled is None:
            raise WebserviceException('No parameters provided to update.', logger=module_logger)

        self._validate_update(None, None, None, None, None, None, None, None, enable_app_insights, None, None,
                              None, None, tags, properties, description, None, None, None, None, None, None,
                              None, None, None, auth_enabled, token_auth_enabled, None, None)

        patch_list = []
        properties = properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        if auth_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/authEnabled', 'value': auth_enabled})
        if token_auth_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/aadAuthEnabled', 'value': token_auth_enabled})
        if enable_app_insights is not None:
            patch_list.append({'op': 'replace', 'path': '/appInsightsEnabled', 'value': enable_app_insights})
        if tags is not None:
            patch_list.append({'op': 'replace', 'path': '/kvTags', 'value': tags})
        if properties is not None:
            for key in properties:
                patch_list.append({'op': 'add', 'path': '/properties/{}'.format(key), 'value': properties[key]})
        if description:
            patch_list.append({'op': 'replace', 'path': '/description', 'value': description})

        self._patch_endpoint_call(headers, params, patch_list)

    def delete_version(self, version_name):
        """Delete a version in an Endpoint.

        :param version_name: The name of the version in an endpoint to delete.
        :type version_name: str
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if version_name is None:
            raise WebserviceException('No name is provided to delete a version in an Endpoint {}'.format(self.name),
                                      logger=module_logger)

        patch_list = []
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}
        patch_list.append({'op': 'remove', 'path': '/versions/{}'.format(version_name)})
        self._patch_endpoint_call(headers, params, patch_list)

    def create_version(self, version_name, autoscale_enabled=None, autoscale_min_replicas=None,
                       autoscale_max_replicas=None, autoscale_refresh_seconds=None, autoscale_target_utilization=None,
                       collect_model_data=None, cpu_cores=None, memory_gb=None,
                       scoring_timeout_ms=None, replica_max_concurrent_requests=None,
                       max_request_wait_time=None, num_replicas=None, tags=None, properties=None, description=None,
                       models=None, inference_config=None, gpu_cores=None, period_seconds=None,
                       initial_delay_seconds=None, timeout_seconds=None, success_threshold=None,
                       failure_threshold=None, traffic_percentile=None, is_default=None, is_control_version_type=None,
                       cpu_cores_limit=None, memory_gb_limit=None):
        """Add a new version in an Endpoint with provided properties.

        :param version_name: The name of the version to add in an endpoint.
        :type version_name: str
        :param autoscale_enabled: Whether or not to enable autoscaling for this version in an Endpoint.
            Defaults to True if ``num_replicas`` is None.
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this version in an
            Endpoint. Defaults to 1
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this version in an
            Endpoint. Defaults to 10
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this version in an Endpoint.
            Defaults to 1
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this version in an Endpoint. Defaults to 70
        :type autoscale_target_utilization: int
        :param collect_model_data: Whether or not to enable model data collection for this version in an Endpoint.
            Defaults to False
        :type collect_model_data: bool
        :param cpu_cores: The number of CPU cores to allocate for this version in an Endpoint. Can be a decimal.
            Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this version in an Endpoint. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param scoring_timeout_ms: A timeout to enforce for scoring calls to this version in an Endpoint.
            Defaults to 60000.
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            version in an Endpoint. Defaults to 1. **Do not change this setting from the default value of 1 unless
            instructed by Microsoft Technical Support or a member of Azure Machine Learning team.**
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error. Defaults to 500.
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this version in an Endpoint. No default, if this
            parameter is not set then the autoscaler is enabled by default.
        :type num_replicas: int
        :param tags: Dictionary of key value tags to give this Endpoint.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Endpoint. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Endpoint.
        :type description: str
        :param models: A list of Model objects to package with the updated service.
        :type models: builtin.list[azureml.core.Model]
        :param inference_config: An InferenceConfig object used to provide the required model deployment properties.
        :type inference_config: azureml.core.model.InferenceConfig
        :param gpu_cores: The number of GPU cores to allocate for this version in an Endpoint. Defaults to 0.
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
            initiated. Defaults to 310.
        :type initial_delay_seconds: int
        :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 2 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try failureThreshold
            times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param traffic_percentile: The amount of traffic the version takes in an endpoint.
        :type traffic_percentile: float
        :param is_default: Whether or not to make this version as default version in an Endpoint.
            Defaults to False.
        :type is_default: bool
        :param is_control_version_type: Whether or not to make this version as control version in an Endpoint.
            Defaults to False.
        :type is_control_version_type: bool
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if version_name is None:
            raise WebserviceException('No name is provided to create a version in an Endpoint {}'.format(self.name),
                                      logger=module_logger)

        self._validate_update(autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                              autoscale_refresh_seconds, autoscale_target_utilization, collect_model_data, cpu_cores,
                              memory_gb, None, scoring_timeout_ms, replica_max_concurrent_requests,
                              max_request_wait_time, num_replicas, tags, properties, description, models,
                              inference_config, gpu_cores, period_seconds, initial_delay_seconds, timeout_seconds,
                              success_threshold, failure_threshold, None, None, None,
                              version_name, traffic_percentile, cpu_cores_limit, memory_gb_limit)

        if models is None:
            models = []

        if inference_config is None:  # No-code deploy.
            environment_image_request = build_and_validate_no_code_environment_image_request(models)
        else:  # Normal inference config route.
            inference_config, has_env = convert_parts_to_environment(version_name, inference_config)

            if not has_env:
                raise WebserviceException('No Environment information is provided.', logger=module_logger)

            environment_image_request = \
                inference_config._build_environment_image_request(self.workspace, [model.id for model in models])

        patch_list = []
        properties = properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        for existing_version_name in self.versions:
            if is_default:
                if existing_version_name != version_name:
                    patch_list.append(
                        {'op': 'replace', 'path': '/versions/{}/isDefault'.format(existing_version_name),
                         'value': False})

            if is_control_version_type:
                if existing_version_name != version_name:
                    patch_list.append(
                        {'op': 'replace', 'path': '/versions/{}/type'.format(existing_version_name),
                         'value': AksEndpoint.VersionType.treatment.value})

        version_deployment_config = AksServiceDeploymentConfiguration(autoscale_enabled, autoscale_min_replicas,
                                                                      autoscale_max_replicas,
                                                                      autoscale_refresh_seconds,
                                                                      autoscale_target_utilization,
                                                                      collect_model_data, None,
                                                                      cpu_cores, memory_gb,
                                                                      None, scoring_timeout_ms,
                                                                      replica_max_concurrent_requests,
                                                                      max_request_wait_time,
                                                                      num_replicas, None, None, tags,
                                                                      properties,
                                                                      description, gpu_cores, period_seconds,
                                                                      initial_delay_seconds,
                                                                      timeout_seconds, success_threshold,
                                                                      failure_threshold, None,
                                                                      None, None,
                                                                      cpu_cores_limit, memory_gb_limit)
        deployment_target = ComputeTarget(self.workspace, self.compute_name)
        version_create_payload = version_deployment_config._build_create_payload(version_name,
                                                                                 environment_image_request,
                                                                                 deployment_target)
        if traffic_percentile is not None:
            version_create_payload['trafficPercentile'] = traffic_percentile
        if is_default:
            version_create_payload['isDefault'] = True
        if is_control_version_type:
            version_create_payload['type'] = AksEndpoint.VersionType.control.value
        patch_list.append({'op': 'add', 'path': '/versions/{}'.format(version_name), 'value': version_create_payload})
        self._patch_endpoint_call(headers, params, patch_list)

    def update_version(self, version_name, autoscale_enabled=None, autoscale_min_replicas=None,
                       autoscale_max_replicas=None, autoscale_refresh_seconds=None, autoscale_target_utilization=None,
                       collect_model_data=None, cpu_cores=None, memory_gb=None,
                       scoring_timeout_ms=None, replica_max_concurrent_requests=None,
                       max_request_wait_time=None, num_replicas=None, tags=None, properties=None, description=None,
                       models=None, inference_config=None, gpu_cores=None, period_seconds=None,
                       initial_delay_seconds=None, timeout_seconds=None, success_threshold=None,
                       failure_threshold=None, traffic_percentile=None, is_default=None, is_control_version_type=None,
                       cpu_cores_limit=None, memory_gb_limit=None):
        """Update an existing version in an Endpoint with provided properties.

        Values left as None will remain unchanged in this version.

        :param version_name: The name of the version in an endpoint.
        :type version_name: str
        :param autoscale_enabled: Whether or not to enable autoscaling for this version in an Endpoint.
            Defaults to True if num_replicas is None.
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this version in an
            Endpoint. Defaults to 1.
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this version in an
            Endpoint. Defaults to 10.
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this version in an Endpoint.
            Defaults to 1
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this version in an Endpoint. Defaults to 70.
        :type autoscale_target_utilization: int
        :param collect_model_data: Whether or not to enable model data collection for this version in an Endpoint.
            Defaults to False.
        :type collect_model_data: bool
        :param cpu_cores: The number of cpu cores to allocate for this version in an Endpoint. Can be a decimal.
            Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this version in an Endpoint. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param scoring_timeout_ms: A timeout to enforce for scoring calls to this version in an Endpoint.
            Defaults to 60000.
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            version in an Endpoint. Defaults to 1. **Do not change this setting from the default value of 1 unless
            instructed by Microsoft Technical Support or a member of Azure Machine Learning team.**
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error. Defaults to 500.
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this version in an Endpoint. No default,
            if this parameter is not set then the autoscaler is enabled by default.
        :type num_replicas: int
        :param tags: Dictionary of key value tags to give this Endpoint.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Endpoint. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Endpoint
        :type description: str
        :param models: A list of Model objects to package with the updated service
        :type models: builtin.list[azureml.core.Model]
        :param inference_config: An InferenceConfig object used to provide the required model deployment properties.
        :type inference_config: azureml.core.model.InferenceConfig
        :param gpu_cores: The number of GPU cores to allocate for this version in an Endpoint. Defaults to 0.
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
            initiated. Defaults to 310.
        :type initial_delay_seconds: int
        :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 2 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try failureThreshold
            times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param traffic_percentile: The amount of traffic the version takes in an endpoint.
        :type traffic_percentile: float
        :param is_default: Whether or not to make this version as default version in an Endpoint.
            Defaults to False.
        :type is_default: bool
        :param is_control_version_type: Whether or not to make this version as control version in an Endpoint.
            Defaults to False.
        :type is_control_version_type: bool
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if version_name is None:
            raise WebserviceException('No name is provided to update a version.', logger=module_logger)

        if autoscale_enabled is None and not autoscale_min_replicas and not autoscale_max_replicas \
                and not autoscale_refresh_seconds and not autoscale_target_utilization and collect_model_data is None \
                and not cpu_cores and not memory_gb and not gpu_cores \
                and not scoring_timeout_ms and not replica_max_concurrent_requests \
                and not max_request_wait_time and not num_replicas and tags is None and properties is None \
                and not description and not period_seconds and not initial_delay_seconds and not timeout_seconds \
                and models is None and inference_config is None and not failure_threshold and not success_threshold \
                and traffic_percentile is None and is_default is None and is_control_version_type is None \
                and cpu_cores_limit is None and memory_gb_limit is None:
            raise WebserviceException('No parameters provided to update a version.', logger=module_logger)

        patch_list = []
        properties = properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}
        for existing_version_name in self.versions:
            if is_default:
                if existing_version_name == version_name:
                    patch_list.append(
                        {'op': 'replace', 'path': '/versions/{}/isDefault'.format(existing_version_name),
                         'value': True})
                else:
                    patch_list.append(
                        {'op': 'replace', 'path': '/versions/{}/isDefault'.format(existing_version_name),
                         'value': False})
            if is_control_version_type:
                if existing_version_name == version_name:
                    patch_list.append(
                        {'op': 'replace', 'path': '/versions/{}/type'.format(existing_version_name),
                         'value': AksEndpoint.VersionType.control.value})
                else:
                    patch_list.append(
                        {'op': 'replace', 'path': '/versions/{}/type'.format(existing_version_name),
                         'value': AksEndpoint.VersionType.treatment.value})

        if inference_config:
            if models is None:
                models = self.versions[version_name].models

            inference_config, has_env = convert_parts_to_environment(version_name, inference_config)

            if not has_env:
                raise WebserviceException('No Environment information is provided.', logger=module_logger)

            environment_image_request = \
                inference_config._build_environment_image_request(self.workspace,
                                                                  [model.id for model in models])
            patch_list.append({'op': 'replace', 'path': '/versions/{}/environmentImageRequest'.format(version_name),
                               'value': environment_image_request})
        elif models is not None:
            # No-code deploy.
            environment_image_request = build_and_validate_no_code_environment_image_request(models)
            patch_list.append({'op': 'replace', 'path': '/versions/{}/environmentImageRequest'.format(version_name),
                               'value': environment_image_request})

        if autoscale_enabled is not None:
            patch_list.append(
                {'op': 'replace', 'path': '/versions/{}/autoScaler/autoscaleEnabled'.format(version_name),
                 'value': autoscale_enabled})
        if autoscale_min_replicas:
            patch_list.append(
                {'op': 'replace', 'path': '/versions/{}/autoScaler/minReplicas'.format(version_name),
                 'value': autoscale_min_replicas})
        if autoscale_max_replicas:
            patch_list.append(
                {'op': 'replace', 'path': '/versions/{}/autoScaler/maxReplicas'.format(version_name),
                 'value': autoscale_max_replicas})
        if autoscale_refresh_seconds:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/autoScaler/refreshPeriodInSeconds'.format(version_name),
                               'value': autoscale_refresh_seconds})
        if autoscale_target_utilization:
            patch_list.append(
                {'op': 'replace', 'path': '/versions/{}/autoScaler/targetUtilization'.format(version_name),
                 'value': autoscale_target_utilization})
        if collect_model_data is not None:
            patch_list.append(
                {'op': 'replace', 'path': '/versions/{}/dataCollection/storageEnabled'.format(version_name),
                 'value': collect_model_data})

        if cpu_cores:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/containerResourceRequirements/cpu'.format(version_name),
                               'value': cpu_cores})

        if cpu_cores_limit:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/containerResourceRequirements/cpuLimit'.format(version_name),
                               'value': cpu_cores_limit})

        if memory_gb:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/containerResourceRequirements/memoryInGB'.format(
                                   version_name),
                               'value': memory_gb})

        if memory_gb_limit:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/containerResourceRequirements/memoryInGBLimit'.format(
                                   version_name),
                               'value': memory_gb_limit})

        if gpu_cores:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/containerResourceRequirements/gpu'.format(version_name),
                               'value': gpu_cores})
        if scoring_timeout_ms:
            patch_list.append({'op': 'replace', 'path': '/versions/{}/scoringTimeoutMs'.format(version_name),
                               'value': scoring_timeout_ms})
        if replica_max_concurrent_requests:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/maxConcurrentRequestsPerContainer'.format(version_name),
                               'value': replica_max_concurrent_requests})
        if max_request_wait_time:
            patch_list.append({'op': 'replace', 'path': '/versions/{}/maxQueueWaitMs'.format(version_name),
                               'value': max_request_wait_time})
        if num_replicas:
            patch_list.append({'op': 'replace', 'path': '/versions/{}/numReplicas'.format(version_name),
                               'value': num_replicas})
        if period_seconds:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/livenessProbeRequirements/periodSeconds'.format(
                                   version_name),
                               'value': period_seconds})
        if initial_delay_seconds:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/livenessProbeRequirements/initialDelaySeconds'.format(
                                   version_name),
                               'value': initial_delay_seconds})
        if timeout_seconds:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/livenessProbeRequirements/timeoutSeconds'.format(
                                   version_name),
                               'value': timeout_seconds})
        if success_threshold:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/livenessProbeRequirements/successThreshold'.format(
                                   version_name),
                               'value': success_threshold})
        if failure_threshold:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/livenessProbeRequirements/failureThreshold'.format(
                                   version_name),
                               'value': failure_threshold})
        if traffic_percentile is not None:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/trafficPercentile'.format(
                                   version_name),
                               'value': traffic_percentile})
        if description:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/description'.format(
                                   version_name),
                               'value': description})
        if tags:
            patch_list.append({'op': 'replace',
                               'path': '/versions/{}/tags'.format(
                                   version_name),
                               'value': tags})

        self._patch_endpoint_call(headers, params, patch_list)

    def _patch_endpoint_call(self, headers, params, patch_list):
        Webservice._check_for_webservice(self.workspace, self.name, self.compute_type,
                                         patch_list, SERVICE_REQUEST_OPERATION_UPDATE)
        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list)

        if resp.status_code == 200:
            self.update_deployment_state()
        elif resp.status_code == 202:
            if 'Operation-Location' in resp.headers:
                operation_location = resp.headers['Operation-Location']
            else:
                raise WebserviceException('Missing response header key: Operation-Location', logger=module_logger)
            create_operation_status_id = operation_location.split('/')[-1]
            base_url = '/'.join(self._mms_endpoint.split('/')[:-2])
            operation_url = base_url + '/operations/{}'.format(create_operation_status_id)
            self._operation_endpoint = operation_url
            self.update_deployment_state()
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    def _validate_update(self, autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                         autoscale_refresh_seconds, autoscale_target_utilization, collect_model_data, cpu_cores,
                         memory_gb, enable_app_insights, scoring_timeout_ms, replica_max_concurrent_requests,
                         max_request_wait_time, num_replicas, tags, properties, description, models,
                         inference_config, gpu_cores, period_seconds, initial_delay_seconds, timeout_seconds,
                         success_threshold, failure_threshold, namespace, auth_enabled, token_auth_enabled,
                         version_name, traffic_percentile, cpu_cores_limit, memory_gb_limit):

        super(AksEndpoint,
              self)._validate_update(None, autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas,
                                     autoscale_refresh_seconds, autoscale_target_utilization, collect_model_data,
                                     cpu_cores, memory_gb, enable_app_insights, scoring_timeout_ms,
                                     replica_max_concurrent_requests, max_request_wait_time, num_replicas, tags,
                                     properties, description, models, inference_config, gpu_cores, period_seconds,
                                     initial_delay_seconds, timeout_seconds, success_threshold, failure_threshold,
                                     namespace, auth_enabled, token_auth_enabled, cpu_cores_limit, memory_gb_limit)

        if version_name is not None:
            webservice_name_validation(version_name)
        error = ""
        if traffic_percentile and (traffic_percentile < 0 or traffic_percentile > 100):
            error += 'Invalid configuration, traffic_percentile cannot be negative or greater than 100.\n'

        if error:
            raise WebserviceException(error, logger=module_logger)

    def serialize(self):
        """Convert this Webservice into a JSON serialized dictionary.

        :return: The JSON representation of this Webservice.
        :rtype: dict
        """
        created_time = self.created_time.isoformat() if self.created_time else None
        updated_time = self.updated_time.isoformat() if self.updated_time else None

        serialized_versions = {}
        for version in self.versions:
            serialized_versions.update({version: self.versions[version].serialize()})
        aks_endpoint_properties = {'name': self.name, 'description': self.description, 'tags': self.tags,
                                   'properties': self.properties, 'state': self.state, 'createdTime': created_time,
                                   'updatedTime': updated_time, 'error': self.error, 'computeName': self.compute_name,
                                   'computeType': self.compute_type, 'workspaceName': self.workspace.name,
                                   'deploymentStatus': self.deployment_status, 'scoringUri': self.scoring_uri,
                                   'authEnabled': self.auth_enabled, 'aadAuthEnabled': self.token_auth_enabled,
                                   'appInsightsEnabled': self.enable_app_insights, 'versions': serialized_versions}
        return aks_endpoint_properties


class AksServiceAccessToken(WebServiceAccessToken):
    """Describes the access token that can be specified in the Authorization header of scoring requests to Webservice.

    :param access_token: The access token.
    :type access_token: str
    :param refresh_after: Time after which access token should be fetched again.
    :type refresh_after: datetime
    :param expiry_on: Expiration time of the access token.
    :type expiry_on: datetime
    :param token_type: The type of access token.
    :type token_type: str
    """


class AutoScaler(object):
    """Defines details for autoscaling configuration of a AksWebservice.

    AutoScaler configuration values are specified using the ``deploy_configuration`` or ``update`` methods
    of the :class:`azureml.core.webservice.aks.AksWebservice` class.

    :var autoscale_enabled: Indicates whether the AutoScaler is enabled or disabled.
    :vartype autoscale_enabled: bool
    :var max_replicas: The maximum number of containers for the AutoScaler to use.
    :vartype max_replicas: int
    :var min_replicas: The minimum number of containers for the AutoScaler to use
    :vartype min_replicas: int
    :var refresh_period_seconds: How often the AutoScaler should attempt to scale the Webservice.
    :vartype refresh_period_seconds: int
    :var target_utilization: The target utilization (in percent out of 100) the AutoScaler should
        attempt to maintain for the Webservice.
    :vartype target_utilization: int

    :param autoscale_enabled: Indicates whether the AutoScaler is enabled or disabled.
    :type autoscale_enabled: bool
    :param max_replicas: The maximum number of containers for the AutoScaler to use.
    :type max_replicas: int
    :param min_replicas: The minimum number of containers for the AutoScaler to use
    :type min_replicas: int
    :param refresh_period_seconds: How often the AutoScaler should attempt to scale the Webservice.
    :type refresh_period_seconds: int
    :param target_utilization: The target utilization (in percent out of 100) the AutoScaler should
        attempt to maintain for the Webservice.
    :type target_utilization: int
    """

    _expected_payload_keys = ['autoscaleEnabled', 'maxReplicas', 'minReplicas', 'refreshPeriodInSeconds',
                              'targetUtilization']

    def __init__(self, autoscale_enabled, max_replicas, min_replicas, refresh_period_seconds, target_utilization):
        """Initialize the AKS AutoScaler.

        :param autoscale_enabled: Indicates whether the AutoScaler is enabled or disabled.
        :type autoscale_enabled: bool
        :param max_replicas: The maximum number of containers for the AutoScaler to use.
        :type max_replicas: int
        :param min_replicas: The minimum number of containers for the AutoScaler to use
        :type min_replicas: int
        :param refresh_period_seconds: How often the AutoScaler should attempt to scale the Webservice.
        :type refresh_period_seconds: int
        :param target_utilization: The target utilization (in percent out of 100) the AutoScaler should
            attempt to maintain for the Webservice.
        :type target_utilization: int
        """
        self.autoscale_enabled = autoscale_enabled
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.refresh_period_seconds = refresh_period_seconds
        self.target_utilization = target_utilization

    def serialize(self):
        """Convert this AutoScaler object into a JSON serialized dictionary.

        :return: The JSON representation of this AutoScaler object.
        :rtype: dict
        """
        return {'autoscaleEnabled': self.autoscale_enabled, 'minReplicas': self.min_replicas,
                'maxReplicas': self.max_replicas, 'refreshPeriodInSeconds': self.refresh_period_seconds,
                'targetUtilization': self.target_utilization}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a AutoScaler object.

        :param payload_obj: A JSON object to convert to a AutoScaler object.
        :type payload_obj: dict
        :return: The AutoScaler representation of the provided JSON object.
        :rtype: azureml.core.webservice.aks.AutoScaler
        """
        for payload_key in AutoScaler._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for autoScaler:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return AutoScaler(payload_obj['autoscaleEnabled'], payload_obj['maxReplicas'], payload_obj['minReplicas'],
                          payload_obj['refreshPeriodInSeconds'], payload_obj['targetUtilization'])


class ContainerResourceRequirements(object):
    """Defines the resource requirements for a container used by the Webservice.

    ContainerResourceRequirement values are specified when deploying or updating a Webervice. For example, use the
    ``deploy_configuration`` or ``update`` methods of the :class:`azureml.core.webservice.aks.AksWebservice` class, or
    the ``create_version``, ``deploy_configuration``, or ``update_version`` methods of
    :class:`azureml.core.webservice.aks.AksEndpoint` class.

    :var cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
    :vartype cpu: float
    :var memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
    :vartype memory_in_gb: float
    :var cpu_limit: The max number of CPU cores this Webservice is allowed to use. Can be a decimal.
    :vartype cpu_limit: float
    :var memory_in_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :vartype memory_in_gb_limit: float

    :param cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
    :type cpu: float
    :param memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
    :type memory_in_gb: float
    :param cpu_limit: The max number of CPU cores this Webservice is allowed to use. Can be a decimal.
    :type cpu_limit: float
    :param memory_in_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :type memory_in_gb_limit: float
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
        :rtype: azureml.core.webservice.aks.ContainerResourceRequirements
        """
        for payload_key in ContainerResourceRequirements._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for ContainerResourceRequirements:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return ContainerResourceRequirements(cpu=payload_obj['cpu'], memory_in_gb=payload_obj['memoryInGB'],
                                             gpu=payload_obj['gpu'], cpu_limit=payload_obj['cpuLimit'],
                                             memory_in_gb_limit=payload_obj['memoryInGBLimit'])


class LivenessProbeRequirements(object):
    """Defines liveness probe time requirements for deployments of the Webservice.

    LivenessProbeRequirements configuration values values are specified when deploying or updating a Webervice.
    For example, use the ``deploy_configuration`` or ``update`` methods of the
    :class:`azureml.core.webservice.aks.AksWebservice` class, or the ``create_version``, ``deploy_configuration``,
    or ``update_version`` methods of the :class:`azureml.core.webservice.aks.AksEndpoint` class.

    :var period_seconds: How often (in seconds) to perform the liveness probe. Defaults to 10 seconds.
        Minimum value is 1.
    :vartype period_seconds: int
    :var initial_delay_seconds: The number of seconds after the container has started before liveness probes are
        initiated.
    :vartype initial_delay_seconds: int
    :var timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 1 second.
        Minimum value is 1.
    :vartype timeout_seconds: int
    :var failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
        ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
    :vartype failure_threshold: int
    :var success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
        after having failed. Defaults to 1. Minimum value is 1.
    :vartype success_threshold: int

    :param period_seconds: How often (in seconds) to perform the liveness probe. Defaults to 10 seconds.
        Minimum value is 1.
    :type period_seconds: int
    :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
        initiated.
    :type initial_delay_seconds: int
    :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 1 second.
        Minimum value is 1.
    :type timeout_seconds: int
    :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
        ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
    :type failure_threshold: int
    :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
        after having failed. Defaults to 1. Minimum value is 1.
    :type success_threshold: int
    """

    _expected_payload_keys = ['periodSeconds', 'initialDelaySeconds', 'timeoutSeconds',
                              'failureThreshold', 'successThreshold']

    def __init__(self, period_seconds, initial_delay_seconds, timeout_seconds, success_threshold, failure_threshold):
        """Initialize the container resource requirements.

        :param period_seconds: How often (in seconds) to perform the liveness probe. Defaults to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
            initiated.
        :type initial_delay_seconds: int
        :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 1 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
            ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        """
        self.period_seconds = period_seconds
        self.timeout_seconds = timeout_seconds
        self.initial_delay_seconds = initial_delay_seconds
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold

    def serialize(self):
        """Convert this LivenessProbeRequirements object into a JSON serialized dictionary.

        :return: The JSON representation of this LivenessProbeRequirements object.
        :rtype: dict
        """
        return {'periodSeconds': self.period_seconds, 'initialDelaySeconds': self.initial_delay_seconds,
                'timeoutSeconds': self.timeout_seconds, 'successThreshold': self.success_threshold,
                'failureThreshold': self.failure_threshold}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a LivenessProbeRequirements object.

        :param payload_obj: A JSON object to convert to a LivenessProbeRequirements object.
        :type payload_obj: dict
        :return: The LivenessProbeRequirements representation of the provided JSON object.
        :rtype: azureml.core.webservice.aks.LivenessProbeRequirements
        """
        if payload_obj is None:
            return LivenessProbeRequirements(period_seconds=10, initial_delay_seconds=310, timeout_seconds=1,
                                             success_threshold=1, failure_threshold=3)
        for payload_key in LivenessProbeRequirements._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for LivenessProbeRequirements:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return LivenessProbeRequirements(payload_obj['periodSeconds'], payload_obj['initialDelaySeconds'],
                                         payload_obj['timeoutSeconds'], payload_obj['successThreshold'],
                                         payload_obj['failureThreshold'])


class DataCollection(object):
    """Defines data collection configuration for an :class:`azureml.core.webservice.aks.AksWebservice`.

    :var event_hub_enabled: Indicates whether event hub is enabled for the Webservice.
    :vartype event_hub_enabled: bool
    :var storage_enabled: Indicates whether data collection storage is enabled for the Webservice.
    :vartype storage_enabled: bool

    :param event_hub_enabled: Indicates whether event hub is enabled for the Webservice.
    :type event_hub_enabled: bool
    :param storage_enabled: Indicates whether data collection storage is enabled for the Webservice.
    :type storage_enabled: bool
    """

    _expected_payload_keys = ['eventHubEnabled', 'storageEnabled']

    def __init__(self, event_hub_enabled, storage_enabled):
        """Intialize the DataCollection object.

        :param event_hub_enabled: Indicates whether event hub is enabled for the Webservice.
        :type event_hub_enabled: bool
        :param storage_enabled: Indicates whether data collection storage is enabled for the Webservice.
        :type storage_enabled: bool
        """
        self.event_hub_enabled = event_hub_enabled
        self.storage_enabled = storage_enabled

    def serialize(self):
        """Convert this DataCollection into a JSON serialized dictionary.

        :return: The JSON representation of this DataCollection object.
        :rtype: dict
        """
        return {'eventHubEnabled': self.event_hub_enabled, 'storageEnabled': self.storage_enabled}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a DataCollection object.

        :param payload_obj: A JSON object to convert to a DataCollection object.
        :type payload_obj: dict
        :return: The DataCollection representation of the provided JSON object.
        :rtype: azureml.core.webservice.aks.DataCollection
        """
        for payload_key in DataCollection._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for DataCollection:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return DataCollection(payload_obj['eventHubEnabled'], payload_obj['storageEnabled'])


class AksServiceDeploymentConfiguration(WebserviceDeploymentConfiguration):
    """Represents a deployment configuration information for a service deployed on Azure Kubernetes Service.

    Create an AksServiceDeploymentConfiguration object using the ``deploy_configuration`` method of the
    :class:`azureml.core.webservice.aks.AksWebservice` class.

    :var autoscale_enabled: Indicates whether to enable autoscaling for this Webservice.
        Defaults to True if ``num_replicas`` is None.
    :vartype autoscale_enabled: bool
    :var autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice.
        Defaults to 1.
    :vartype autoscale_min_replicas: int
    :var autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice.
        Defaults to 10
    :vartype autoscale_max_replicas: int
    :var autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice.
        Defaults to 1.
    :vartype autoscale_refresh_seconds: int
    :var autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
        attempt to maintain for this Webservice. Defaults to 70.
    :vartype autoscale_target_utilization: int
    :var collect_model_data: Whether or not to enable model data collection for this Webservice.
        Defaults to False.
    :vartype collect_model_data: bool
    :var auth_enabled: Whether or not to enable auth for this Webservice. Defaults to True.
    :vartype auth_enabled: bool
    :var cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
    :vartype cpu_cores: float
    :var memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        Defaults to 0.5
    :vartype memory_gb: float
    :var enable_app_insights: Whether or not to enable Application Insights logging for this Webservice.
        Defaults to False
    :vartype enable_app_insights: bool
    :var scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice. Defaults to 60000.
    :vartype scoring_timeout_ms: int
    :var replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
        Webservice. Defaults to 1. **Do not change this setting from the default value of 1 unless instructed by
        Microsoft Technical Support or a member of Azure Machine Learning team.**
    :vartype replica_max_concurrent_requests: int
    :var max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
        before returning a 503 error. Defaults to 500.
    :vartype max_request_wait_time: int
    :var num_replicas: The number of containers to allocate for this Webservice. No default, if this parameter
        is not set then the autoscaler is enabled by default.
    :vartype num_replicas: int
    :var primary_key: A primary auth key to use for this Webservice.
    :vartype primary_key: str
    :var secondary_key: A secondary auth key to use for this Webservice.
    :vartype secondary_key: str
    :var azureml.core.webservice.AksServiceDeploymentConfiguration.tags: Dictionary of key value tags to give this
        Webservice.
    :vartype tags: dict[str, str]
    :var azureml.core.webservice.AksServiceDeploymentConfiguration.properties: Dictionary of key value properties to
        give this Webservice. These properties cannot be changed after deployment, however new key value pairs can
        be added.
    :vartype properties: dict[str, str]
    :var azureml.core.webservice.AksServiceDeploymentConfiguration.description: A description to give this Webservice.
    :vartype description: str
    :var gpu_cores: The number of GPU cores to allocate for this Webservice. Defaults to 0.
    :vartype gpu_cores: int
    :var period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
        Minimum value is 1.
    :vartype period_seconds: int
    :var initial_delay_seconds: Number of seconds after the container has started before liveness probes are
        initiated. Defaults to 310.
    :vartype initial_delay_seconds: int
    :var timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 2 second.
        Minimum value is 1.
    :vartype timeout_seconds: int
    :var success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
        after having failed. Defaults to 1. Minimum value is 1.
    :vartype success_threshold: int
    :var failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try ``failureThreshold``
        times before giving up. Defaults to 3. Minimum value is 1.
    :vartype failure_threshold: int
    :var azureml.core.webservice.AksServiceDeploymentConfiguration.namespace: The Kubernetes namespace in which to
        deploy this Webservice: up to 63 lowercase alphanumeric ('a'-'z', '0'-'9') and hyphen ('-') characters. The
        first and last characters cannot be hyphens.
    :vartype namespace: str
    :var token_auth_enabled: Whether or not to enable Azure Active Directory auth for this Webservice. If this is
            enabled, users can access this Webservice by fetching access token using their Azure Active Directory
            credentials. Defaults to False.
    :vartype token_auth_enabled: bool
    :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
    :vartype cpu_cores_limit: float
    :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :vartype memory_gb_limit: float

    :param autoscale_enabled: Indicates whether to enable autoscaling for this Webservice.
        Defaults to True if ``num_replicas`` is None.
    :type autoscale_enabled: bool
    :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice.
        Defaults to 1.
    :type autoscale_min_replicas: int
    :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice.
        Defaults to 10
    :type autoscale_max_replicas: int
    :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice.
        Defaults to 1.
    :type autoscale_refresh_seconds: int
    :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
        attempt to maintain for this Webservice. Defaults to 70.
    :type autoscale_target_utilization: int
    :param collect_model_data: Whether or not to enable model data collection for this Webservice.
        Defaults to False.
    :type collect_model_data: bool
    :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to True.
    :type auth_enabled: bool
    :param cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
    :type cpu_cores: float
    :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        Defaults to 0.5
    :type memory_gb: float
    :param enable_app_insights: Whether or not to enable Application Insights logging for this Webservice.
        Defaults to False
    :type enable_app_insights: bool
    :param scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice. Defaults to 60000.
    :type scoring_timeout_ms: int
    :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
        Webservice. Defaults to 1. **Do not change this setting from the default value of 1 unless instructed by
        Microsoft Technical Support or a member of Azure Machine Learning team.**
    :type replica_max_concurrent_requests: int
    :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
        before returning a 503 error. Defaults to 500.
    :type max_request_wait_time: int
    :param num_replicas: The number of containers to allocate for this Webservice. No default, if this parameter
        is not set then the autoscaler is enabled by default.
    :type num_replicas: int
    :param primary_key: A primary auth key to use for this Webservice.
    :type primary_key: str
    :param secondary_key: A secondary auth key to use for this Webservice.
    :type secondary_key: str
    :param tags: Dictionary of key value tags to give this Webservice.
    :type tags: dict[str, str]
    :param properties: Dictionary of key value properties to give this Webservice. These properties cannot
        be changed after deployment, however new key value pairs can be added.
    :type properties: dict[str, str]
    :param description: A description to give this Webservice.
    :type description: str
    :param gpu_cores: The number of GPU cores to allocate for this Webservice. Defaults to 0.
    :type gpu_cores: int
    :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
        Minimum value is 1.
    :type period_seconds: int
    :param initial_delay_seconds: Number of seconds after the container has started before liveness probes are
        initiated. Defaults to 310.
    :type initial_delay_seconds: int
    :param timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 2 second.
        Minimum value is 1.
    :type timeout_seconds: int
    :param success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
        after having failed. Defaults to 1. Minimum value is 1.
    :type success_threshold: int
    :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
        ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
    :type failure_threshold: int
    :param namespace: The Kubernetes namespace in which to deploy this Webservice: up to 63 lowercase alphanumeric
        ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
    :type namespace: str
    :param token_auth_enabled: Whether or not to enable Azure Active Directory auth for this Webservice. If this is
            enabled, users can access this Webservice by fetching access token using their Azure Active Directory
            credentials. Defaults to False.
    :type token_auth_enabled: bool
    :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
    :vartype cpu_cores_limit: float
    :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :vartype memory_gb_limit: float
    :param blobfuse_enabled: Whether or not to enable blobfuse for model downloading for this Webservice.
        Defaults to True
    :type blobfuse_enabled: bool
    """

    def __init__(self, autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas, autoscale_refresh_seconds,
                 autoscale_target_utilization, collect_model_data, auth_enabled, cpu_cores,
                 memory_gb, enable_app_insights, scoring_timeout_ms, replica_max_concurrent_requests,
                 max_request_wait_time, num_replicas, primary_key, secondary_key, tags, properties, description,
                 gpu_cores, period_seconds, initial_delay_seconds, timeout_seconds, success_threshold,
                 failure_threshold, namespace, token_auth_enabled, compute_target_name, cpu_cores_limit,
                 memory_gb_limit, blobfuse_enabled=None):
        """Initialize a configuration object for deploying to an AKS compute target.

        :param autoscale_enabled: Indicates whether to enable autoscaling for this Webservice.
            Defaults to True if ``num_replicas`` is None.
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice.
            Defaults to 1.
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice.
            Defaults to 10
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice.
            Defaults to 1.
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this Webservice. Defaults to 70.
        :type autoscale_target_utilization: int
        :param collect_model_data: Whether or not to enable model data collection for this Webservice.
            Defaults to False.
        :type collect_model_data: bool
        :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to True.
        :type auth_enabled: bool
        :param cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param enable_app_insights: Whether or not to enable Application Insights logging for this Webservice.
            Defaults to False
        :type enable_app_insights: bool
        :param scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice. Defaults to 60000.
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            Webservice. Defaults to 1. **Do not change this setting from the default value of 1 unless instructed by
            Microsoft Technical Support or a member of Azure Machine Learning team.**
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error. Defaults to 500.
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this Webservice. No default, if this parameter
            is not set then the autoscaler is enabled by default.
        :type num_replicas: int
        :param primary_key: A primary auth key to use for this Webservice.
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Webservice.
        :type secondary_key: str
        :param tags: Dictionary of key value tags to give this Webservice.
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Webservice. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Webservice.
        :type description: str
        :param gpu_cores: The number of GPU cores to allocate for this Webservice. Defaults to 0.
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: Number of seconds after the container has started before liveness probes are
            initiated. Defaults to 310.
        :type initial_delay_seconds: int
        :param timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 2 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
            ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param namespace: The Kubernetes namespace in which to deploy this Webservice: up to 63 lowercase alphanumeric
            ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
        :type namespace: str
        :param token_auth_enabled: Whether or not to enable Azure Active Directory auth for this Webservice. If this is
                enabled, users can access this Webservice by fetching access token using their Azure Active Directory
                credentials. Defaults to False.
        :type token_auth_enabled: bool
        :param compute_target_name: The name of the compute target to deploy to
        :type compute_target_name: str
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :param blobfuse_enabled: Whether or not to enable blobfuse for model downloading for this Webservice.
            Defaults to True
        :type blobfuse_enabled: bool
        :return: A configuration object to use when deploying a Webservice object.
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        super(AksServiceDeploymentConfiguration, self).__init__(AksWebservice, description, tags, properties,
                                                                primary_key, secondary_key)
        self.autoscale_enabled = autoscale_enabled
        self.autoscale_min_replicas = autoscale_min_replicas
        self.autoscale_max_replicas = autoscale_max_replicas
        self.autoscale_refresh_seconds = autoscale_refresh_seconds
        self.autoscale_target_utilization = autoscale_target_utilization
        self.collect_model_data = collect_model_data
        self.auth_enabled = auth_enabled
        self.cpu_cores = cpu_cores
        self.cpu_cores_limit = cpu_cores_limit
        self.memory_gb = memory_gb
        self.memory_gb_limit = memory_gb_limit
        self.gpu_cores = gpu_cores
        self.enable_app_insights = enable_app_insights
        self.scoring_timeout_ms = scoring_timeout_ms
        self.replica_max_concurrent_requests = replica_max_concurrent_requests
        self.max_request_wait_time = max_request_wait_time
        self.num_replicas = num_replicas
        self.period_seconds = period_seconds
        self.initial_delay_seconds = initial_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.namespace = namespace
        self.token_auth_enabled = token_auth_enabled
        self.blobfuse_enabled = blobfuse_enabled
        self.compute_target_name = compute_target_name
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Will raise a WebserviceException if validation fails.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        error = ""
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            error += 'Invalid configuration, cpu_cores must be greater than zero.\n'
        if self.cpu_cores is not None and self.cpu_cores_limit is not None and self.cpu_cores_limit < self.cpu_cores:
            error += 'Invalid configuration, cpu_cores_limit must be greater than or equal to cpu_cores.\n'
        if self.memory_gb is not None and self.memory_gb <= 0:
            error += 'Invalid configuration, memory_gb must be greater than zero.\n'
        if self.memory_gb is not None and self.memory_gb_limit is not None and self.memory_gb_limit < self.memory_gb:
            error += 'Invalid configuration, memory_gb_limit must be greater than or equal to memory_gb.\n'
        if self.gpu_cores is not None and self.gpu_cores < 0:
            error += 'Invalid configuration, gpu_cores must be greater than or equal to zero.\n'
        if self.period_seconds is not None and self.period_seconds <= 0:
            error += 'Invalid configuration, period_seconds must be greater than zero.\n'
        if self.initial_delay_seconds is not None and self.initial_delay_seconds <= 0:
            error += 'Invalid configuration, initial_delay_seconds must be greater than zero.\n'
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            error += 'Invalid configuration, timeout_seconds must be greater than zero.\n'
        if self.success_threshold is not None and self.success_threshold <= 0:
            error += 'Invalid configuration, success_threshold must be greater than zero.\n'
        if self.failure_threshold is not None and self.failure_threshold <= 0:
            error += 'Invalid configuration, failure_threshold must be greater than zero.\n'
        if self.namespace and not re.match(NAMESPACE_REGEX, self.namespace):
            error += 'Invalid configuration, namespace must be a valid Kubernetes namespace. ' \
                     'Regex for validation is ' + NAMESPACE_REGEX + '\n'
        if self.scoring_timeout_ms is not None and self.scoring_timeout_ms <= 0:
            error += 'Invalid configuration, scoring_timeout_ms must be greater than zero.\n'
        if self.replica_max_concurrent_requests is not None and self.replica_max_concurrent_requests <= 0:
            error += 'Invalid configuration, replica_max_concurrent_requests must be greater than zero.\n'
        if self.max_request_wait_time is not None and self.max_request_wait_time <= 0:
            error += 'Invalid configuration, max_request_wait_time must be greater than zero.\n'
        if self.num_replicas is not None and self.num_replicas <= 0:
            error += 'Invalid configuration, num_replicas must be greater than zero.\n'
        if self.autoscale_enabled:
            if self.num_replicas:
                error += 'Invalid configuration, autoscale enabled and num_replicas provided.\n'
            if self.autoscale_min_replicas is not None and self.autoscale_min_replicas <= 0:
                error += 'Invalid configuration, autoscale_min_replicas must be greater than zero.\n'
            if self.autoscale_max_replicas is not None and self.autoscale_max_replicas <= 0:
                error += 'Invalid configuration, autoscale_max_replicas must be greater than zero.\n'
            if self.autoscale_min_replicas and self.autoscale_max_replicas and \
                    self.autoscale_min_replicas > self.autoscale_max_replicas:
                error += 'Invalid configuration, autoscale_min_replicas cannot be greater than ' \
                         'autoscale_max_replicas.\n'
            if self.autoscale_refresh_seconds is not None and self.autoscale_refresh_seconds <= 0:
                error += 'Invalid configuration, autoscale_refresh_seconds must be greater than zero.\n'
            if self.autoscale_target_utilization is not None and self.autoscale_target_utilization <= 0:
                error += 'Invalid configuration, autoscale_target_utilization must be greater than zero.\n'
        else:
            if self.autoscale_enabled is False and not self.num_replicas:
                error += 'Invalid configuration, autoscale disabled but num_replicas not provided.\n'
            if self.autoscale_min_replicas:
                error += 'Invalid configuration, autoscale_min_replicas provided without enabling autoscaling.\n'
            if self.autoscale_max_replicas:
                error += 'Invalid configuration, autoscale_max_replicas provided without enabling autoscaling.\n'
            if self.autoscale_refresh_seconds:
                error += 'Invalid configuration, autoscale_refresh_seconds provided without enabling autoscaling.\n'
            if self.autoscale_target_utilization:
                error += 'Invalid configuration, autoscale_target_utilization provided without enabling autoscaling.\n'
        if self.token_auth_enabled and self.auth_enabled:
            error += "Invalid configuration, auth_enabled and token_auth_enabled cannot both be true.\n"

        if error:
            raise WebserviceException(error, logger=module_logger)

    def print_deploy_configuration(self):
        """Print the deployment configuration."""
        deploy_config = []
        if self.cpu_cores:
            deploy_config.append('CPU requirement: {}'.format(self.cpu_cores))
        if self.gpu_cores:
            deploy_config.append('GPU requirement: {}'.format(self.gpu_cores))
        if self.memory_gb:
            deploy_config.append('Memory requirement: {}GB'.format(self.memory_gb))
        if self.num_replicas:
            deploy_config.append('Number of replica: {}'.format(self.num_replicas))

        if len(deploy_config) > 0:
            print(', '.join(deploy_config))

    def _build_create_payload(self, name, environment_image_request, deployment_target=None, overwrite=False):
        import copy
        from azureml._model_management._util import aks_specific_service_create_payload_template
        json_payload = copy.deepcopy(aks_specific_service_create_payload_template)
        base_payload = super(AksServiceDeploymentConfiguration,
                             self)._build_base_create_payload(name, environment_image_request)

        json_payload['numReplicas'] = self.num_replicas
        if self.collect_model_data:
            json_payload['dataCollection']['storageEnabled'] = self.collect_model_data
        else:
            del(json_payload['dataCollection'])
        if self.enable_app_insights is not None:
            json_payload['appInsightsEnabled'] = self.enable_app_insights
        else:
            del(json_payload['appInsightsEnabled'])
        if self.blobfuse_enabled is not None:
            json_payload['storageInitEnabled'] = not self.blobfuse_enabled
        else:
            del(json_payload['storageInitEnabled'])
        if self.autoscale_enabled is not None:
            json_payload['autoScaler']['autoscaleEnabled'] = self.autoscale_enabled
            json_payload['autoScaler']['minReplicas'] = self.autoscale_min_replicas
            json_payload['autoScaler']['maxReplicas'] = self.autoscale_max_replicas
            json_payload['autoScaler']['targetUtilization'] = self.autoscale_target_utilization
            json_payload['autoScaler']['refreshPeriodInSeconds'] = self.autoscale_refresh_seconds
        else:
            del(json_payload['autoScaler'])
        json_payload['containerResourceRequirements']['cpu'] = self.cpu_cores
        json_payload['containerResourceRequirements']['cpuLimit'] = self.cpu_cores_limit
        json_payload['containerResourceRequirements']['memoryInGB'] = self.memory_gb
        json_payload['containerResourceRequirements']['memoryInGBLimit'] = self.memory_gb_limit
        json_payload['containerResourceRequirements']['gpu'] = self.gpu_cores
        json_payload['maxConcurrentRequestsPerContainer'] = self.replica_max_concurrent_requests
        json_payload['maxQueueWaitMs'] = self.max_request_wait_time
        json_payload['namespace'] = self.namespace
        json_payload['scoringTimeoutMs'] = self.scoring_timeout_ms
        if self.auth_enabled is not None:
            json_payload['authEnabled'] = self.auth_enabled
        else:
            del(json_payload['authEnabled'])
        if self.token_auth_enabled is not None:
            json_payload['aadAuthEnabled'] = self.token_auth_enabled
        else:
            del(json_payload['aadAuthEnabled'])
        json_payload['livenessProbeRequirements']['periodSeconds'] = self.period_seconds
        json_payload['livenessProbeRequirements']['initialDelaySeconds'] = self.initial_delay_seconds
        json_payload['livenessProbeRequirements']['timeoutSeconds'] = self.timeout_seconds
        json_payload['livenessProbeRequirements']['failureThreshold'] = self.failure_threshold
        json_payload['livenessProbeRequirements']['successThreshold'] = self.success_threshold

        if overwrite:
            json_payload['overwrite'] = overwrite
        else:
            del (json_payload['overwrite'])

        if deployment_target is not None:
            json_payload['computeName'] = deployment_target.name

        json_payload.update(base_payload)

        return json_payload


class AksEndpointDeploymentConfiguration(AksServiceDeploymentConfiguration):
    """Represents deployment configuration information for a service deployed on Azure Kubernetes Service.

    Create an AksEndpointDeploymentConfiguration object using the ``deploy_configuration`` method of the
    :class:`azureml.core.webservice.aks.AksEndpoint` class.

    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.autoscale_enabled: Whether or not to enable
        autoscaling for this Webservice. Defaults to True if ``num_replicas`` is None.
    :vartype autoscale_enabled: bool
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.autoscale_min_replicas: The minimum number of
        containers to use when autoscaling this Webservice. Defaults to 1.
    :vartype autoscale_min_replicas: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.autoscale_max_replicas: The maximum number of
        containers to use when autoscaling this Webservice. Defaults to 10.
    :vartype autoscale_max_replicas: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.autoscale_refresh_seconds: How often the
        autoscaler should attempt to scale this Webservice. Defaults to 1.
    :vartype autoscale_refresh_seconds: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.autoscale_target_utilization: The target
        utilization (in percent out of 100) the autoscaler should attempt to maintain for this Webservice.
        Defaults to 70.
    :vartype autoscale_target_utilization: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.collect_model_data: Whether or not to enable
        model data collection for this Webservice. Defaults to False.
    :vartype collect_model_data: bool
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.auth_enabled: Whether or not to enable auth
        for this Webservice. Defaults to True.
    :vartype auth_enabled: bool
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.cpu_cores: The number of cpu cores to
        allocate for this Webservice. Can be a decimal. Defaults to 0.1
    :vartype cpu_cores: float
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.memory_gb: The amount of memory (in GB) to
        allocate for this Webservice. Can be a decimal. Defaults to 0.5
    :vartype memory_gb: float
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.enable_app_insights: Whether or not to
        enable Application Insights logging for this Webservice. Defaults to False.
    :vartype enable_app_insights: bool
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.scoring_timeout_ms: A timeout to enforce for
        scoring calls to this Webservice. Defaults to 60000.
    :vartype scoring_timeout_ms: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.replica_max_concurrent_requests: The number
        of maximum concurrent requests per replica to allow for this Webservice. Defaults to 1.
        **Do not change this setting from the default value of 1 unless instructed by
        Microsoft Technical Support or a member of Azure Machine Learning team.**
    :vartype replica_max_concurrent_requests: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.max_request_wait_time: The maximum amount of
        time a request will stay in the queue (in milliseconds) before returning a 503 error. Defaults to 500.
    :vartype max_request_wait_time: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.num_replicas: The number of containers to
        allocate for this Webservice. No default, if this parameter is not set then the autoscaler is enabled by
        default.
    :vartype num_replicas: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.primary_key: A primary auth key to use for
        this Webservice
    :vartype primary_key: str
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.secondary_key: A secondary auth key to use
        for this Webservice
    :vartype secondary_key: str
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.tags: Dictionary of key value tags to give
        this Webservice
    :vartype tags: dict[str, str]
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.properties: Dictionary of key value
        properties to give this Webservice. These properties cannot be changed after deployment, however new key
        value pairs can be added.
    :vartype properties: dict[str, str]
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.description: A description to give this
        Webservice.
    :vartype description: str
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.gpu_cores: The number of GPU cores to
        allocate for this Webservice. Defaults to 0.
    :vartype gpu_cores: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.period_seconds: How often (in seconds) to
        perform the liveness probe. Default to 10 seconds. Minimum value is 1.
    :vartype period_seconds: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.initial_delay_seconds: The number of seconds
        after the container has started before liveness probes are initiated. Defaults to 310.
    :vartype initial_delay_seconds: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.timeout_seconds: The number of seconds after
        which the liveness probe times out. Defaults to 2 second. Minimum value is 1.
    :vartype timeout_seconds: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.success_threshold: The minimum consecutive
        successes for the liveness probe to be considered successful after having failed. Defaults to 1. Minimum
        value is 1.
    :vartype success_threshold: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.failure_threshold: When a Pod starts and the
        liveness probe fails, Kubernetes will try ``failureThreshold`` times before giving up. Defaults to 3. Minimum
        value is 1.
    :vartype failure_threshold: int
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.namespace: The Kubernetes namespace in which
        to deploy this Webservice: up to 63 lowercase alphanumeric ('a'-'z', '0'-'9') and hyphen ('-') characters.
        The first and last characters cannot be hyphens.
    :vartype namespace: str
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.token_auth_enabled: Whether or not to enable
        Azure Active Directory auth for this Webservice. If this is enabled, users can access this Webservice by
        fetching access token using their Azure Active Directory credentials. Defaults to False.
    :vartype token_auth_enabled: bool
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.version_name: The name of the version in an
        endpoint.
    :vartype version_name: str
    :var azureml.core.webservice.aks.AksEndpointDeploymentConfiguration.traffic_percentile: The amount of traffic the
        version takes in an endpoint.
    :vartype traffic_percentile: float

    :param autoscale_enabled: Whether or not to enable autoscaling for this Webservice.
        Defaults to True if ``num_replicas`` is None.
    :type autoscale_enabled: bool
    :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice.
        Defaults to 1.
    :type autoscale_min_replicas: int
    :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice.
        Defaults to 10.
    :type autoscale_max_replicas: int
    :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice.
        Defaults to 1.
    :type autoscale_refresh_seconds: int
    :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
        attempt to maintain for this Webservice. Defaults to 70.
    :type autoscale_target_utilization: int
    :param collect_model_data: Whether or not to enable model data collection for this Webservice.
        Defaults to False.
    :type collect_model_data: bool
    :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to True.
    :type auth_enabled: bool
    :param cpu_cores: The number of cpu cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
    :type cpu_cores: float
    :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        Defaults to 0.5
    :type memory_gb: float
    :param enable_app_insights: Whether or not to enable Application Insights logging for this Webservice.
        Defaults to False.
    :type enable_app_insights: bool
    :param scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice. Defaults to 60000.
    :type scoring_timeout_ms: int
    :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
        Webservice. Defaults to 1. **Do not change this setting from the default value of 1 unless instructed by
        Microsoft Technical Support or a member of Azure Machine Learning team.**
    :type replica_max_concurrent_requests: int
    :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
        before returning a 503 error. Defaults to 500.
    :type max_request_wait_time: int
    :param num_replicas: The number of containers to allocate for this Webservice. No default, if this parameter
        is not set then the autoscaler is enabled by default.
    :type num_replicas: int
    :param primary_key: A primary auth key to use for this Webservice
    :type primary_key: str
    :param secondary_key: A secondary auth key to use for this Webservice
    :type secondary_key: str
    :param tags: Dictionary of key value tags to give this Webservice
    :type tags: dict[str, str]
    :param properties: Dictionary of key value properties to give this Webservice. These properties cannot
        be changed after deployment, however new key value pairs can be added.
    :type properties: dict[str, str]
    :param description: A description to give this Webservice.
    :type description: str
    :param gpu_cores: The number of GPU cores to allocate for this Webservice. Defaults to 0.
    :type gpu_cores: int
    :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
        Minimum value is 1.
    :type period_seconds: int
    :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
        initiated. Defaults to 310.
    :type initial_delay_seconds: int
    :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 2 second.
        Minimum value is 1.
    :type timeout_seconds: int
    :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
        after having failed. Defaults to 1. Minimum value is 1.
    :type success_threshold: int
    :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
        ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
    :type failure_threshold: int
    :param namespace: The Kubernetes namespace in which to deploy this Webservice: up to 63 lowercase alphanumeric
        ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
    :type namespace: str
    :param token_auth_enabled: Whether or not to enable Azure Active Directory auth for this Webservice. If this is
            enabled, users can access this Webservice by fetching access token using their Azure Active Directory
            credentials. Defaults to False.
    :type token_auth_enabled: bool
    :param version_name: The name of the version in an endpoint.
    :type version_name: str
    :param traffic_percentile: The amount of traffic the version takes in an endpoint.
    :type traffic_percentile: float
    :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
    :type cpu_cores_limit: float
    :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :type memory_gb_limit: float
    """

    def __init__(self, autoscale_enabled, autoscale_min_replicas, autoscale_max_replicas, autoscale_refresh_seconds,
                 autoscale_target_utilization, collect_model_data, auth_enabled, cpu_cores, memory_gb,
                 enable_app_insights, scoring_timeout_ms, replica_max_concurrent_requests,
                 max_request_wait_time, num_replicas, primary_key, secondary_key, tags, properties, description,
                 gpu_cores, period_seconds, initial_delay_seconds, timeout_seconds, success_threshold,
                 failure_threshold, namespace, token_auth_enabled, version_name, traffic_percentile,
                 compute_target_name, cpu_cores_limit, memory_gb_limit):
        """Initialize a configuration object for deploying an Endpoint to an AKS compute target.

        :param autoscale_enabled: Whether or not to enable autoscaling for this Webservice.
            Defaults to True if ``num_replicas`` is None.
        :type autoscale_enabled: bool
        :param autoscale_min_replicas: The minimum number of containers to use when autoscaling this Webservice.
            Defaults to 1.
        :type autoscale_min_replicas: int
        :param autoscale_max_replicas: The maximum number of containers to use when autoscaling this Webservice.
            Defaults to 10.
        :type autoscale_max_replicas: int
        :param autoscale_refresh_seconds: How often the autoscaler should attempt to scale this Webservice.
            Defaults to 1.
        :type autoscale_refresh_seconds: int
        :param autoscale_target_utilization: The target utilization (in percent out of 100) the autoscaler should
            attempt to maintain for this Webservice. Defaults to 70.
        :type autoscale_target_utilization: int
        :param collect_model_data: Whether or not to enable model data collection for this Webservice.
            Defaults to False.
        :type collect_model_data: bool
        :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to True.
        :type auth_enabled: bool
        :param cpu_cores: The number of cpu cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param enable_app_insights: Whether or not to enable Application Insights logging for this Webservice.
            Defaults to False.
        :type enable_app_insights: bool
        :param scoring_timeout_ms: A timeout to enforce for scoring calls to this Webservice. Defaults to 60000.
        :type scoring_timeout_ms: int
        :param replica_max_concurrent_requests: The number of maximum concurrent requests per replica to allow for this
            Webservice. Defaults to 1. **Do not change this setting from the default value of 1 unless instructed by
            Microsoft Technical Support or a member of Azure Machine Learning team.**
        :type replica_max_concurrent_requests: int
        :param max_request_wait_time: The maximum amount of time a request will stay in the queue (in milliseconds)
            before returning a 503 error. Defaults to 500.
        :type max_request_wait_time: int
        :param num_replicas: The number of containers to allocate for this Webservice. No default, if this parameter
            is not set then the autoscaler is enabled by default.
        :type num_replicas: int
        :param primary_key: A primary auth key to use for this Webservice
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Webservice
        :type secondary_key: str
        :param tags: Dictionary of key value tags to give this Webservice
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give this Webservice. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Webservice.
        :type description: str
        :param gpu_cores: The number of GPU cores to allocate for this Webservice. Defaults to 0.
        :type gpu_cores: int
        :param period_seconds: How often (in seconds) to perform the liveness probe. Default to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
            initiated. Defaults to 310.
        :type initial_delay_seconds: int
        :param timeout_seconds: The number of seconds after which the liveness probe times out. Defaults to 2 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param success_threshold: The minimum consecutive successes for the liveness probe to be considered successful
            after having failed. Defaults to 1. Minimum value is 1.
        :type success_threshold: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
            ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param namespace: The Kubernetes namespace in which to deploy this Webservice: up to 63 lowercase alphanumeric
            ('a'-'z', '0'-'9') and hyphen ('-') characters. The first and last characters cannot be hyphens.
        :type namespace: str
        :param token_auth_enabled: Whether or not to enable Azure Active Directory auth for this Webservice. If this is
                enabled, users can access this Webservice by fetching access token using their Azure Active Directory
                credentials. Defaults to False.
        :type token_auth_enabled: bool
        :param version_name: The name of the version in an endpoint.
        :type version_name: str
        :param traffic_percentile: The amount of traffic the version takes in an endpoint.
        :type traffic_percentile: float
        :param compute_target_name: The name of the compute target to deploy to
        :type compute_target_name: str
        :param cpu_cores_limit: The max number of cpu cores this Webservice is allowed to use. Can be a decimal.
        :type cpu_cores_limit: float
        :param memory_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
        :type memory_gb_limit: float
        :return: A configuration object to use when deploying an Endpoint object.
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        super(AksEndpointDeploymentConfiguration, self).__init__(autoscale_enabled, autoscale_min_replicas,
                                                                 autoscale_max_replicas, autoscale_refresh_seconds,
                                                                 autoscale_target_utilization, collect_model_data,
                                                                 auth_enabled, cpu_cores, memory_gb,
                                                                 enable_app_insights, scoring_timeout_ms,
                                                                 replica_max_concurrent_requests,
                                                                 max_request_wait_time, num_replicas, primary_key,
                                                                 secondary_key, tags, properties, description,
                                                                 gpu_cores, period_seconds, initial_delay_seconds,
                                                                 timeout_seconds, success_threshold, failure_threshold,
                                                                 namespace, token_auth_enabled, compute_target_name,
                                                                 cpu_cores_limit, memory_gb_limit)
        self._webservice_type = AksEndpoint
        self.version_name = version_name
        self.traffic_percentile = traffic_percentile
        self.validate_endpoint_configuration()

    def validate_endpoint_configuration(self):
        """Check that the specified configuration values are valid.

        Will raise a WebserviceException if validation fails.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        error = ""
        if self.version_name:
            webservice_name_validation(self.version_name)
        if self.traffic_percentile and (self.traffic_percentile < 0 or self.traffic_percentile > 100):
            error += 'Invalid configuration, traffic_percentile cannot be negative or greater than 100.\n'

        if error:
            raise WebserviceException(error, logger=module_logger)

    def _build_create_payload(self, name, environment_image_request, overwrite=False):
        from azureml._model_management._util import aks_specific_endpoint_create_payload_template
        json_payload = copy.deepcopy(aks_specific_endpoint_create_payload_template)
        if self.version_name is None:
            self.version_name = name

        version_payload = super(AksEndpointDeploymentConfiguration,
                                self)._build_create_payload(self.version_name, environment_image_request)

        # update the version payload
        version_payload['trafficPercentile'] = self.traffic_percentile

        json_payload['name'] = name
        json_payload['description'] = self.description
        json_payload['kvTags'] = self.tags

        properties = self.properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        json_payload['properties'] = properties

        if self.primary_key:
            json_payload['keys']['primaryKey'] = self.primary_key
            json_payload['keys']['secondaryKey'] = self.secondary_key

        json_payload['computeType'] = AksEndpoint._webservice_type
        if self.enable_app_insights is not None:
            json_payload['appInsightsEnabled'] = self.enable_app_insights
        else:
            del(json_payload['appInsightsEnabled'])
        if self.namespace is not None:
            json_payload['namespace'] = self.namespace
        else:
            del(json_payload['namespace'])
        if self.auth_enabled is not None:
            json_payload['authEnabled'] = self.auth_enabled
        else:
            del(json_payload['authEnabled'])
        if self.token_auth_enabled is not None:
            json_payload['aadAuthEnabled'] = self.token_auth_enabled
        else:
            del(json_payload['aadAuthEnabled'])

        if overwrite:
            json_payload['overwrite'] = overwrite
        else:
            del (json_payload['overwrite'])

        json_payload['versions'] = {self.version_name: version_payload}
        return json_payload
