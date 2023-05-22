# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for deploying machine learning models as web service endpoints on Azure Container Instances.

Azure Container Instances (ACI) is recommended for scenarios that can operate in isolated containers,
including simple applications, task automation, and build jobs. For more information about when to use ACI,
see [Deploy a model to Azure Container
Instances](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-container-instance).
"""

import json
import logging
import os
from azureml._base_sdk_common.tracking import global_tracking_info_registry
from azureml._model_management._constants import MMS_SYNC_TIMEOUT_SECONDS
from azureml._model_management._constants import ACI_WEBSERVICE_TYPE
from azureml._model_management._constants import WEBSERVICE_SWAGGER_PATH
from azureml._model_management._constants import SERVICE_REQUEST_OPERATION_UPDATE
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import build_and_validate_no_code_environment_image_request
from azureml._model_management._util import convert_parts_to_environment
from azureml._restclient.clientbase import ClientBase
from azureml.core.environment import Environment
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.webservice.webservice import WebserviceDeploymentConfiguration
from azureml.exceptions import WebserviceException

module_logger = logging.getLogger(__name__)


class AciWebservice(Webservice):
    """Represents a machine learning model deployed as a web service endpoint on Azure Container Instances.

    A deployed service is created from a model, script, and associated files. The resulting web service
    is a load-balanced, HTTP endpoint with a REST API. You can send data to this API and receive the
    prediction returned by the model.

    For more information, see `Deploy a model to Azure Container
    Instances <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-container-instance>`__.

    .. remarks::

        The recommended deployment pattern is to create a deployment configuration object with the
        ``deploy_configuration`` method and then use it with the ``deploy`` method of the
        :class:`azureml.core.model.Model` class as shown below.

        .. code-block:: python

            from azureml.core.model import InferenceConfig
            from azureml.core.webservice import AciWebservice


            service_name = 'my-custom-env-service'

            inference_config = InferenceConfig(entry_script='score.py', environment=environment)
            aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

            service = Model.deploy(workspace=ws,
                                   name=service_name,
                                   models=[model],
                                   inference_config=inference_config,
                                   deployment_config=aci_config,
                                   overwrite=True)
            service.wait_for_deployment(show_output=True)

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/deploy-to-cloud/model-register-and-deploy.ipynb


        There are a number of ways to deploy a model as a webservice,
        including with the:

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

        The *Variables* section lists attributes of a local representation of the cloud AciWebservice object. These
        variables should be considered read-only. Changing their values will not be reflected in the corresponding
        cloud object.

    :var enable_app_insights: Whether or not AppInsights logging is enabled for the Webservice.
    :vartype enable_app_insights: bool
    :var cname: The cname for the Webservice.
    :vartype cname: str
    :var container_resource_requirements: The container resource requirements for the Webservice.
    :vartype container_resource_requirements: azureml.core.webservice.aci.ContainerResourceRequirements
    :var encryption_properties: The encryption properties for the Webservice.
    :vartype encryption_properties: azureml.core.webservice.aci.EncryptionProperties
    :var vnet_configuration: The virtual network properties for the Webservice, configuration should be
                            created and provided by user.
    :vartype vnet_configuration: azureml.core.webservice.aci.VnetConfiguration
    :var azureml.core.webservice.AciWebservice.location: The location the Webservice is deployed to.
    :vartype azureml.core.webservice.AciWebservice.location: str
    :var public_ip: The public ip address of the Webservice.
    :vartype public_ip: str
    :var azureml.core.webservice.AciWebservice.scoring_uri: The scoring endpoint for the Webservice
    :vartype azureml.core.webservice.AciWebservice.scoring_uri: str
    :var ssl_enabled: Whether or not SSL is enabled for the Webservice
    :vartype ssl_enabled: bool
    :var public_fqdn: The public FQDN for the Webservice
    :vartype public_fqdn: str
    :var environment: The Environment object that was used to create the Webservice
    :vartype environment: azureml.core.Environment
    :var azureml.core.webservice.AciWebservice.models: A list of Models deployed to the Webservice
    :vartype azureml.core.webservice.AciWebservice.models: builtin.list[azureml.core.Model]
    :var azureml.core.webservice.AciWebservice.swagger_uri: The swagger endpoint for the Webservice
    :vartype azureml.core.webservice.AciWebservice.swagger_uri: str
    """

    _expected_payload_keys = Webservice._expected_payload_keys + \
        ['appInsightsEnabled', 'authEnabled', 'cname', 'containerResourceRequirements',
         'location', 'publicIp', 'scoringUri', 'sslCertificate', 'sslEnabled', 'sslKey']
    _webservice_type = ACI_WEBSERVICE_TYPE

    def _initialize(self, workspace, obj_dict):
        """Initialize the Webservice instance.

        This is used because the constructor is used as a getter.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        # Validate obj_dict with _expected_payload_keys
        AciWebservice._validate_get_payload(obj_dict)

        # Initialize common Webservice attributes
        super(AciWebservice, self)._initialize(workspace, obj_dict)

        # Initialize expected ACI specific attributes
        self.enable_app_insights = obj_dict.get('appInsightsEnabled')
        self.cname = obj_dict.get('cname')
        self.container_resource_requirements = \
            ContainerResourceRequirements.deserialize(obj_dict.get('containerResourceRequirements'))
        self.encryption_properties = \
            EncryptionProperties.deserialize(obj_dict.get('encryptionProperties'))
        self.vnet_configuration = \
            VnetConfiguration.deserialize(obj_dict.get('vnetConfiguration'))
        self.location = obj_dict.get('location')
        self.public_ip = obj_dict.get('publicIp')
        self.scoring_uri = obj_dict.get('scoringUri')
        self.ssl_certificate = obj_dict.get('sslCertificate')
        self.ssl_enabled = obj_dict.get('sslEnabled')
        self.ssl_key = obj_dict.get('sslKey')
        self.public_fqdn = obj_dict.get('publicFqdn')
        env_image_request = obj_dict.get('environmentImageRequest')
        env_dict = env_image_request.get('environment') if env_image_request else None
        self.environment = Environment._deserialize_and_add_to_object(env_dict) if env_dict else None
        models = obj_dict.get('models')
        self.models = [Model.deserialize(workspace, model_payload) for model_payload in models] if models else []

        # Initialize other ACI utility attributes
        self.swagger_uri = '/'.join(self.scoring_uri.split('/')[:-1]) + WEBSERVICE_SWAGGER_PATH \
            if self.scoring_uri else None
        self._model_config_map = obj_dict.get('modelConfigMap')

    def __repr__(self):
        """Return the string representation of the AciWebservice object.

        :return: String representation of the AciWebservice object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def deploy_configuration(cpu_cores=None, memory_gb=None, tags=None, properties=None, description=None,
                             location=None, auth_enabled=None, ssl_enabled=None, enable_app_insights=None,
                             ssl_cert_pem_file=None, ssl_key_pem_file=None, ssl_cname=None, dns_name_label=None,
                             primary_key=None, secondary_key=None, collect_model_data=None,
                             cmk_vault_base_url=None, cmk_key_name=None, cmk_key_version=None,
                             vnet_name=None, subnet_name=None):
        """Create a configuration object for deploying an AciWebservice.

        :param cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param tags: A dictionary of key value tags to give this Webservice.
        :type tags: dict[str, str]
        :param properties: A dictionary of key value properties to give this Webservice. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Webservice.
        :type description: str
        :param location: The Azure region to deploy this Webservice to. If not specified the Workspace location will
            be used. For more details on available regions, see `Products by
            region <https://azure.microsoft.com/global-infrastructure/services/?products=container-instances>`_.
        :type location: str
        :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to False.
        :type auth_enabled: bool
        :param ssl_enabled: Whether or not to enable SSL for this Webservice. Defaults to False.
        :type ssl_enabled: bool
        :param enable_app_insights: Whether or not to enable AppInsights for this Webservice. Defaults to False.
        :type enable_app_insights: bool
        :param ssl_cert_pem_file: The cert file needed if SSL is enabled.
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: The key file needed if SSL is enabled.
        :type ssl_key_pem_file: str
        :param ssl_cname: The cname for if SSL is enabled.
        :type ssl_cname: str
        :param dns_name_label: The DNS name label for the scoring endpoint.
            If not specified a unique DNS name label will be generated for the scoring endpoint.
        :type dns_name_label: str
        :param primary_key: A primary auth key to use for this Webservice.
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Webservice.
        :type secondary_key: str
        :param collect_model_data: Whether or not to enabled model data collection for the Webservice.
        :type collect_model_data: bool
        :param cmk_vault_base_url: customer managed key vault base url
        :type cmk_vault_base_url: str
        :param cmk_key_name: customer managed key name.
        :type cmk_key_name: str
        :param cmk_key_version: customer managed key version.
        :type cmk_key_version: str
        :param vnet_name: virtual network name.
        :type vnet_name: str
        :param subnet_name: subnet name within virtual network.
        :type subnet_name: str
        :return: A configuration object to use when deploying a Webservice object.
        :rtype: azureml.core.webservice.aci.AciServiceDeploymentConfiguration
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        config = AciServiceDeploymentConfiguration(cpu_cores=cpu_cores, memory_gb=memory_gb, tags=tags,
                                                   properties=properties, description=description,
                                                   location=location, auth_enabled=auth_enabled,
                                                   ssl_enabled=ssl_enabled, enable_app_insights=enable_app_insights,
                                                   ssl_cert_pem_file=ssl_cert_pem_file,
                                                   ssl_key_pem_file=ssl_key_pem_file,
                                                   ssl_cname=ssl_cname, dns_name_label=dns_name_label,
                                                   primary_key=primary_key, secondary_key=secondary_key,
                                                   collect_model_data=collect_model_data,
                                                   cmk_vault_base_url=cmk_vault_base_url,
                                                   cmk_key_name=cmk_key_name,
                                                   cmk_key_version=cmk_key_version,
                                                   vnet_name=vnet_name, subnet_name=subnet_name)
        return config

    @staticmethod
    def _deploy(workspace, name, image, deployment_config, overwrite):  # pragma: no cover
        """Deploy the Webservice.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param image:
        :type image: azureml.core.Image
        :param deployment_config:
        :type deployment_config: AciServiceDeploymentConfiguration | None
        :param overwrite:
        :type overwrite: bool
        :return:
        :rtype: AciWebservice
        """
        if not deployment_config:
            deployment_config = AciWebservice.deploy_configuration()
        elif not isinstance(deployment_config, AciServiceDeploymentConfiguration):
            raise WebserviceException('Error, provided deployment configuration must be of type '
                                      'AciServiceDeploymentConfiguration in order to deploy an ACI service.',
                                      logger=module_logger)
        deployment_config.validate_image(image)
        create_payload = AciWebservice._build_create_payload(name, image, deployment_config, overwrite)
        return Webservice._deploy_webservice(workspace, name, create_payload, overwrite, AciWebservice)

    @staticmethod
    def _build_create_payload(name, image, deploy_config, overwrite):  # pragma: no cover
        """Construct the payload used to create this Webservice.

        :param name:
        :type name: str
        :param image:
        :type image: azureml.core.Image
        :param deploy_config:
        :type deploy_config: azureml.core.compute.AciServiceDeploymentConfiguration
        :return:
        :rtype: dict
        """
        import copy
        from azureml._model_management._util import aci_service_payload_template
        json_payload = copy.deepcopy(aci_service_payload_template)
        json_payload['name'] = name
        json_payload['imageId'] = image.id
        json_payload['kvTags'] = deploy_config.tags
        json_payload['description'] = deploy_config.description
        json_payload['containerResourceRequirements']['cpu'] = deploy_config.cpu_cores
        json_payload['containerResourceRequirements']['memoryInGB'] = deploy_config.memory_gb
        json_payload['location'] = deploy_config.location

        properties = deploy_config.properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        json_payload['properties'] = properties

        if deploy_config.auth_enabled is None:
            del (json_payload['authEnabled'])
        else:
            json_payload['authEnabled'] = deploy_config.auth_enabled
        if deploy_config.ssl_enabled is None:
            del (json_payload['sslEnabled'])
        else:
            json_payload['sslEnabled'] = deploy_config.ssl_enabled
        if deploy_config.enable_app_insights is None:
            del (json_payload['appInsightsEnabled'])
        else:
            json_payload['appInsightsEnabled'] = deploy_config.enable_app_insights
        if deploy_config.dns_name_label:
            json_payload['dnsNameLabel'] = deploy_config.dns_name_label
        else:
            del (json_payload['dnsNameLabel'])
        try:
            with open(deploy_config.ssl_cert_pem_file, 'r') as cert_file:
                cert_data = cert_file.read()
            json_payload['sslCertificate'] = cert_data
        except Exception:
            del (json_payload['sslCertificate'])
        try:
            with open(deploy_config.ssl_key_pem_file, 'r') as key_file:
                key_data = key_file.read()
            json_payload['sslKey'] = key_data
        except Exception:
            del (json_payload['sslKey'])
        if deploy_config.ssl_cname is None:
            del (json_payload['cname'])
        else:
            json_payload['cname'] = deploy_config.ssl_cname

        encryption_properties = {}
        vnet_configuration = {}

        # All 3 propertis will either be all null or all valid
        # validation is done in aciDeploymentConfiguration
        if deploy_config.cmk_vault_base_url is not None:
            encryption_properties['vaultBaseUrl'] = deploy_config.cmk_vault_base_url
            encryption_properties['keyName'] = deploy_config.cmk_key_name
            encryption_properties['keyVersion'] = deploy_config.cmk_key_version
            json_payload['encryptionProperties'] = encryption_properties

        if deploy_config.vnet_name is not None:
            vnet_configuration['vnetName'] = deploy_config.vnet_name
            vnet_configuration['subnetName'] = deploy_config.subnet_name
            json_payload['vnetConfiguration'] = vnet_configuration

        if overwrite:
            json_payload['overwrite'] = overwrite
        else:
            del (json_payload['overwrite'])
        return json_payload

    def run(self, input_data):
        """Call this Webservice with the provided input.

        :param input_data: The input to call the Webservice with.
        :type input_data: varies
        :return: The result of calling the Webservice.
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

    def update(self, image=None, tags=None, properties=None, description=None, auth_enabled=None, ssl_enabled=None,
               ssl_cert_pem_file=None, ssl_key_pem_file=None, ssl_cname=None, enable_app_insights=None, models=None,
               inference_config=None):
        """Update the Webservice with provided properties.

        Values left as None will remain unchanged in this Webservice.

        :param image: A new Image to deploy to the Webservice.
        :type image: azureml.core.Image
        :param tags: A dictionary of key value tags to give this Webservice. Will replace existing tags.
        :type tags: dict[str, str]
        :param properties: A dictionary of key value properties to add to existing properties dictionary.
        :type properties: dict[str, str]
        :param description: A description to give this Webservice.
        :type description: str
        :param auth_enabled: Enable or disable auth for this Webservice.
        :type auth_enabled: bool
        :param ssl_enabled: Whether or not to enable SSL for this Webservice.
        :type ssl_enabled: bool
        :param ssl_cert_pem_file: The cert file needed if SSL is enabled.
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: The key file needed if SSL is enabled.
        :type ssl_key_pem_file: str
        :param ssl_cname: The cname for if SSL is enabled.
        :type ssl_cname: str
        :param enable_app_insights: Whether or not to enable AppInsights for this Webservice.
        :type enable_app_insights: bool
        :param models: A list of Model objects to package into the updated service.
        :type models: builtin.list[azureml.core.Model]
        :param inference_config: An InferenceConfig object used to provide the required model deployment properties.
        :type inference_config: azureml.core.model.InferenceConfig
        :return:
        :rtype: None
        """
        if not image and tags is None and properties is None and not description and auth_enabled is None \
           and ssl_enabled is None and not ssl_cert_pem_file and not ssl_key_pem_file and not ssl_cname \
           and enable_app_insights is None and models is None and inference_config is None:
            raise WebserviceException('No parameters provided to update.', logger=module_logger)

        self._validate_update(image, tags, properties, description, auth_enabled, ssl_enabled, ssl_cert_pem_file,
                              ssl_key_pem_file, ssl_cname, enable_app_insights, models, inference_config)

        cert_data = ""
        key_data = ""
        if ssl_cert_pem_file:
            try:
                with open(ssl_cert_pem_file, 'r') as cert_file:
                    cert_data = cert_file.read()
            except (IOError, OSError) as exc:
                raise WebserviceException("Error while reading ssl information:\n{}".format(exc),
                                          logger=module_logger)
        if ssl_key_pem_file:
            try:
                with open(ssl_key_pem_file, 'r') as key_file:
                    key_data = key_file.read()
            except (IOError, OSError) as exc:
                raise WebserviceException("Error while reading ssl information:\n{}".format(exc),
                                          logger=module_logger)

        patch_list = []

        if inference_config:
            if models is None:
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

        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        properties = properties or {}
        properties.update(global_tracking_info_registry.gather_all())

        if image:
            patch_list.append({'op': 'replace', 'path': '/imageId', 'value': image.id})
        if tags is not None:
            patch_list.append({'op': 'replace', 'path': '/kvTags', 'value': tags})
        if properties is not None:
            for key in properties:
                patch_list.append({'op': 'add', 'path': '/properties/{}'.format(key), 'value': properties[key]})
        if description:
            patch_list.append({'op': 'replace', 'path': '/description', 'value': description})
        if auth_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/authEnabled', 'value': auth_enabled})
        if ssl_enabled is not None:
            patch_list.append({'op': 'replace', 'path': '/sslEnabled', 'value': ssl_enabled})
        if ssl_cert_pem_file:
            patch_list.append({'op': 'replace', 'path': '/sslCertificate', 'value': cert_data})
        if ssl_key_pem_file:
            patch_list.append({'op': 'replace', 'path': '/sslKey', 'value': key_data})
        if ssl_cname:
            patch_list.append({'op': 'replace', 'path': '/cname', 'value': ssl_cname})
        if enable_app_insights is not None:
            patch_list.append({'op': 'replace', 'path': '/appInsightsEnabled', 'value': enable_app_insights})

        Webservice._check_for_webservice(self.workspace, self.name, self.compute_type,
                                         patch_list, SERVICE_REQUEST_OPERATION_UPDATE)
        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list, timeout=MMS_SYNC_TIMEOUT_SECONDS)

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

    def _validate_update(self, image, tags, properties, description, auth_enabled, ssl_enabled, ssl_cert_pem_file,
                         ssl_key_pem_file, ssl_cname, enable_app_insights, models, inference_config):
        error = ""
        if image and self.environment:  # pragma: no cover
            error += 'Error, unable to use Image object to update Webservice created with Environment object.\n'
        if inference_config and inference_config.environment and self.image:  # pragma: no cover
            error += 'Error, unable to use Environment object to update Webservice created with Image object.\n'
        if image and inference_config:  # pragma: no cover
            error += 'Error, unable to pass both an Image object and an InferenceConfig object to update.\n'
        if inference_config is None and models and \
                (len(models) != 1 or models[0].model_framework not in Model._SUPPORTED_FRAMEWORKS_FOR_NO_CODE_DEPLOY):
            error += 'Error, both "models" and "inference_config" inputs must be provided in order ' \
                     'to update the models.\n'
        if (ssl_cert_pem_file or ssl_key_pem_file) and not ssl_enabled and not self.ssl_enabled:
            error += 'Error, SSL must be enabled in order to update SSL cert/key.\n'
        if ssl_cert_pem_file and not os.path.exists(ssl_cert_pem_file):
            error += 'Error, unable to find ssl_cert_pem_file at provided path: {}\n'.format(ssl_cert_pem_file)
        if ssl_key_pem_file and not os.path.exists(ssl_key_pem_file):
            error += 'Error, unable to find ssl_key_pem_file at provided path: {}\n'.format(ssl_key_pem_file)

        if error:
            raise WebserviceException(error, logger=module_logger)

    def add_tags(self, tags):
        """Add key value pairs to this Webservice's tags dictionary.

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

        :param tags: The list of keys to remove.
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

        print('Webservice add properties operation complete.')

    def serialize(self):
        """Convert this Webservice into a JSON serialized dictionary.

        :return: The JSON representation of this Webservice object.
        :rtype: dict
        """
        properties = super(AciWebservice, self).serialize()
        container_resource_requirements = self.container_resource_requirements.serialize() \
            if self.container_resource_requirements else None
        encryption_properties = self.encryption_properties.serialize() \
            if self.encryption_properties else None
        vnet_configuration = self.vnet_configuration.serialize() \
            if self.vnet_configuration else None
        env_details = Environment._serialize_to_dict(self.environment) if self.environment else None
        model_details = [model.serialize() for model in self.models] if self.models else None
        aci_properties = {'containerResourceRequirements': container_resource_requirements, 'imageId': self.image_id,
                          'scoringUri': self.scoring_uri, 'location': self.location,
                          'authEnabled': self.auth_enabled, 'sslEnabled': self.ssl_enabled,
                          'appInsightsEnabled': self.enable_app_insights, 'sslCertificate': self.ssl_certificate,
                          'sslKey': self.ssl_key, 'cname': self.cname, 'publicIp': self.public_ip,
                          'publicFqdn': self.public_fqdn, 'environmentDetails': env_details,
                          'modelDetails': model_details, 'encryptionProperties': encryption_properties,
                          'vnetConfiguration': vnet_configuration}
        properties.update(aci_properties)
        return properties

    def get_token(self):
        """
        Retrieve auth token for this Webservice, scoped to the current user.

        .. note::
            Not implemented.

        :return: The auth token for this Webservice and when it should be refreshed after.
        :rtype: str, datetime
        :raises: azureml.exceptions.NotImplementedError
        """
        raise NotImplementedError("ACI webservices do not support Token Authentication.")


class ContainerResourceRequirements(object):
    """Defines the resource requirements for a container used by the Webservice.

    To specify ContainerResourceRequirements values, you will typically use the ``deploy_configuration`` method of the
    :class:`azureml.core.webservice.aci.AciWebservice` class.

    :var cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
    :vartype cpu: float
    :var memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
    :vartype memory_in_gb: float

    :param cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
    :type cpu: float
    :param memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
    :type memory_in_gb: float
    """

    _expected_payload_keys = ['cpu', 'memoryInGB']

    def __init__(self, cpu, memory_in_gb):
        """Initialize the container resource requirements.

        :param cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
        :type cpu: float
        :param memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        :type memory_in_gb: float
        """
        self.cpu = cpu
        self.memory_in_gb = memory_in_gb

    def serialize(self):
        """Convert this ContainerResourceRequirements object into a JSON serialized dictionary.

        :return: The JSON representation of this ContainerResourceRequirements object.
        :rtype: dict
        """
        return {'cpu': self.cpu, 'memoryInGB': self.memory_in_gb}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a ContainerResourceRequirements object.

        :param payload_obj: A JSON object to convert to a ContainerResourceRequirements object.
        :type payload_obj: dict
        :return: The ContainerResourceRequirements representation of the provided JSON object.
        :rtype: azureml.core.webservice.aci.ContainerResourceRequirements
        """
        for payload_key in ContainerResourceRequirements._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for containerResourceReservation:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return ContainerResourceRequirements(payload_obj['cpu'], payload_obj['memoryInGB'])


class EncryptionProperties(object):
    """Defines the encryption properties for a container used by the Webservice.

    To specify EncryptionProperties values, you will typically use the ``deploy_configuration`` method
    of the :class:`azureml.core.webservice.aci.AciWebservice` class.

    :var cmk_vault_base_url: Customer managed key vault base url.
    :vartype cmk_vault_base_url: str
    :var cmk_key_name: Customer managed key name.
    :vartype cmk_key_name: str
    :var cmk_key_version: Customer managed key version.
    :vartype cmk_key_version: str
    """

    _expected_payload_keys = ['vaultBaseUrl', 'keyName', 'keyVersion']

    def __init__(self, cmk_vault_base_url, cmk_key_name, cmk_key_version):
        """Initialize encryption properties.

        :param cmk_vault_base_url: customer managed key vault base url.
        :type cmk_vault_base_url: str
        :param cmk_key_name: customer managed key name.
        :type cmk_key_name: str
        :param cmk_key_version: customer managed key version.
        :type cmk_key_version: str
        """
        self.cmk_vault_base_url = cmk_vault_base_url
        self.cmk_key_name = cmk_key_name
        self.cmk_key_version = cmk_key_version

    def serialize(self):
        """Convert this EncryptionProperties object into a JSON serialized dictionary.

        :return: The JSON representation of this EncryptionProperties object.
        :rtype: dict
        """
        return {'vaultBaseUrl': self.cmk_vault_base_url,
                'keyName': self.cmk_key_name,
                'keyVersion': self.cmk_key_version}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a EncryptionProperties object.

        :param payload_obj: A JSON object to convert to a EncryptionProperties object.
        :type payload_obj: dict
        :return: The EncryptionProperties representation of the provided JSON object.
        :rtype: azureml.core.webservice.aci.EncryptionProperties
        """
        if payload_obj is None:
            return None
        for payload_key in EncryptionProperties._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for EncryptionProperties:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return EncryptionProperties(payload_obj['vaultBaseUrl'], payload_obj['keyName'], payload_obj['keyVersion'])


class VnetConfiguration(object):
    """Defines the Virtual network configuration for a container used by the Webservice.

    To specify VnetConfiguration values, you will typically use the ``deploy_configuration`` method
    of the :class:`azureml.core.webservice.aci.AciWebservice` class.

    :var vnet_name: Virtual network name.
    :vartype vnet_name: str
    :var subnet_name: Subnet name within virtual network.
    :vartype subnet_name: str
    """

    _expected_payload_keys = ['vnetName', 'subnetName']

    def __init__(self, vnet_name, subnet_name):
        """Initialize encryption properties.

        :param vnet_name: Virtual network name.
        :type vnet_name: str
        :param subnet_name: Subnet name within virtual network.
        :type subnet_name: str
        """
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name

    def serialize(self):
        """Convert this VnetConfiguration object into a JSON serialized dictionary.

        :return: The JSON representation of this VnetConfiguration object.
        :rtype: dict
        """
        return {'vnetName': self.vnet_name,
                'subnetName': self.subnet_name}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a VnetConfiguration object.

        :param payload_obj: A JSON object to convert to a VnetConfiguration object.
        :type payload_obj: dict
        :return: The VnetConfiguration representation of the provided JSON object.
        :rtype: azureml.core.webservice.aci.VnetConfiguration
        """
        if payload_obj is None:
            return None
        for payload_key in VnetConfiguration._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for VnetConfiguration:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return VnetConfiguration(payload_obj['vnetName'], payload_obj['subnetName'])


class AciServiceDeploymentConfiguration(WebserviceDeploymentConfiguration):
    """Represents deployment configuration information for a service deployed on Azure Container Instances.

    Create an AciServiceDeploymentConfiguration object using the ``deploy_configuration`` method of the
    :class:`azureml.core.webservice.aci.AciWebservice` class.

    :var cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
    :vartype cpu_cores: float
    :var memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        Defaults to 0.5
    :vartype memory_gb: float
    :var azureml.core.webservice.AciServiceDeploymentConfiguration.tags: A dictionary of key value tags to give this
        Webservice.
    :vartype tags: dict[str, str]
    :var azureml.core.webservice.AciServiceDeploymentConfiguration.properties: A dictionary of key value properties
        to give this Webservice. These properties cannot be changed after deployment, however new key value pairs
        can be added.
    :vartype properties: dict[str, str]
    :var azureml.core.webservice.AciServiceDeploymentConfiguration.description: A description to give this Webservice.
    :vartype description: str
    :var azureml.core.webservice.AciServiceDeploymentConfiguration.location: The Azure region to deploy this
        Webservice to. If not specified, the Workspace location will be used. For more details on available regions,
        see `Products by region
        <https://azure.microsoft.com/global-infrastructure/services/?products=container-instances>`_.
    :vartype location: str
    :var auth_enabled: Whether or not to enable auth for this Webservice. Defaults to False.
    :vartype auth_enabled: bool
    :var ssl_enabled: Whether or not to enable SSL for this Webservice. Defaults to False.
    :vartype ssl_enabled: bool
    :var enable_app_insights: Whether or not to enable AppInsights for this Webservice. Defaults to False.
    :vartype enable_app_insights: bool
    :var ssl_cert_pem_file: The cert file needed if SSL is enabled.
    :vartype ssl_cert_pem_file: str
    :var ssl_key_pem_file: The key file needed if SSL is enabled.
    :vartype ssl_key_pem_file: str
    :var ssl_cname: The cname for if SSL is enabled.
    :vartype ssl_cname: str
    :var dns_name_label: The DNS name label for the scoring endpoint.
        If not specified a unique DNS name label will be generated for the scoring endpoint.
    :vartype dns_name_label: str
    :var primary_key: A primary auth key to use for this Webservice.
    :vartype primary_key: str
    :var secondary_key: A secondary auth key to use for this Webservice.
    :vartype secondary_key: str
    :var collect_model_data: Whether or not to enabled model data collection for the Webservice.
    :vartype collect_model_data: bool

    :param cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
    :type cpu_cores: float
    :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
        Defaults to 0.5
    :type memory_gb: float
    :param tags: A dictionary of key value tags to give this Webservice.
    :type tags: dict[str, str]
    :param properties: A dictionary of key value properties to give this Webservice. These properties cannot
        be changed after deployment, however new key value pairs can be added.
    :type properties: dict[str, str]
    :param description: A description to give this Webservice.
    :type description: str
    :param location: The Azure region to deploy this Webservice to. If not specified, the Workspace location will
        be used. For more details on available regions, see `Products by
        region <https://azure.microsoft.com/global-infrastructure/services/?products=container-instances>`__.
    :type location: str
    :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to False.
    :type auth_enabled: bool
    :param ssl_enabled: Whether or not to enable SSL for this Webservice. Defaults to False.
    :type ssl_enabled: bool
    :param enable_app_insights: Whether or not to enable AppInsights for this Webservice. Defaults to False.
    :type enable_app_insights: bool
    :param ssl_cert_pem_file: The cert file needed if SSL is enabled.
    :type ssl_cert_pem_file: str
    :param ssl_key_pem_file: The key file needed if SSL is enabled.
    :type ssl_key_pem_file: str
    :param ssl_cname: The cname for if SSL is enabled.
    :type ssl_cname: str
    :param dns_name_label: The DNS name label for the scoring endpoint.
        If not specified a unique DNS name label will be generated for the scoring endpoint.
    :type dns_name_label: str
    :param primary_key: A primary auth key to use for this Webservice.
    :type primary_key: str
    :param secondary_key: A secondary auth key to use for this Webservice.
    :type secondary_key: str
    :param collect_model_data: Whether or not to enable model data collection for this Webservice.
        Defaults to False
    :type collect_model_data: bool
    :param cmk_vault_base_url: customer managed key vault base url
    :type cmk_vault_base_url: str
    :param cmk_key_name: customer managed key name.
    :type cmk_key_name: str
    :param cmk_key_version: customer managed key version.
    :type cmk_key_version: str
    :param vnet_name: virtual network name.
    :type vnet_name: str
    :param subnet_name: subnet name within virtual network.
    :type subnet_name: str
    """

    _webservice_type = AciWebservice

    def __init__(self, cpu_cores=None, memory_gb=None, tags=None, properties=None, description=None, location=None,
                 auth_enabled=None, ssl_enabled=None, enable_app_insights=None, ssl_cert_pem_file=None,
                 ssl_key_pem_file=None, ssl_cname=None, dns_name_label=None,
                 primary_key=None, secondary_key=None, collect_model_data=None,
                 cmk_vault_base_url=None, cmk_key_name=None, cmk_key_version=None,
                 vnet_name=None, subnet_name=None):
        """Create a configuration object for deploying an ACI Webservice.

        :param cpu_cores: The number of CPU cores to allocate for this Webservice. Can be a decimal. Defaults to 0.1
        :type cpu_cores: float
        :param memory_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
            Defaults to 0.5
        :type memory_gb: float
        :param tags: A dictionary of key value tags to give this Webservice.
        :type tags: dict[str, str]
        :param properties: A dictionary of key value properties to give this Webservice. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param description: A description to give this Webservice.
        :type description: str
        :param location: The Azure region to deploy this Webservice to. If not specified, the Workspace location will
            be used. For more details on available regions, see `Products by
            region <https://azure.microsoft.com/global-infrastructure/services/?products=container-instances>`__.
        :type location: str
        :param auth_enabled: Whether or not to enable auth for this Webservice. Defaults to False.
        :type auth_enabled: bool
        :param ssl_enabled: Whether or not to enable SSL for this Webservice. Defaults to False.
        :type ssl_enabled: bool
        :param enable_app_insights: Whether or not to enable AppInsights for this Webservice. Defaults to False.
        :type enable_app_insights: bool
        :param ssl_cert_pem_file: The cert file needed if SSL is enabled.
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: The key file needed if SSL is enabled.
        :type ssl_key_pem_file: str
        :param ssl_cname: The cname for if SSL is enabled.
        :type ssl_cname: str
        :param dns_name_label: The DNS name label for the scoring endpoint.
            If not specified a unique DNS name label will be generated for the scoring endpoint.
        :type dns_name_label: str
        :param primary_key: A primary auth key to use for this Webservice.
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Webservice.
        :type secondary_key: str
        :param collect_model_data: Whether or not to enable model data collection for this Webservice.
            Defaults to False
        :type collect_model_data: bool
        :param cmk_vault_base_url: customer managed key vault base url
        :type cmk_vault_base_url: str
        :param cmk_key_name: customer managed key name.
        :type cmk_key_name: str
        :param cmk_key_version: customer managed key version.
        :type cmk_key_version: str
        :param vnet_name: virtual network name.
        :type vnet_name: str
        :param subnet_name: subnet name within virtual network.
        :type subnet_name: str
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        super(AciServiceDeploymentConfiguration, self).__init__(AciWebservice, description, tags, properties,
                                                                primary_key, secondary_key, location)
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.auth_enabled = auth_enabled
        self.ssl_enabled = ssl_enabled
        self.enable_app_insights = enable_app_insights
        self.ssl_cert_pem_file = ssl_cert_pem_file
        self.ssl_key_pem_file = ssl_key_pem_file
        self.ssl_cname = ssl_cname
        self.dns_name_label = dns_name_label
        self.collect_model_data = collect_model_data
        self.cmk_vault_base_url = cmk_vault_base_url
        self.cmk_key_name = cmk_key_name
        self.cmk_key_version = cmk_key_version
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Will raise a :class:`azureml.exceptions.WebserviceException` if validation fails.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            raise WebserviceException('Invalid configuration, cpu_cores must be positive.', logger=module_logger)
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise WebserviceException('Invalid configuration, memory_gb must be positive.', logger=module_logger)
        if self.ssl_enabled:
            if not self.ssl_cert_pem_file or not self.ssl_key_pem_file or not self.ssl_cname:
                raise WebserviceException('SSL is enabled, you must provide a SSL certificate, key, and cname.',
                                          logger=module_logger)
            if not os.path.exists(self.ssl_cert_pem_file):
                raise WebserviceException('Error, unable to find SSL cert pem file provided paths:\n'
                                          '{}'.format(self.ssl_cert_pem_file))
            if not os.path.exists(self.ssl_key_pem_file):
                raise WebserviceException('Error, unable to find SSL key pem file provided paths:\n'
                                          '{}'.format(self.ssl_key_pem_file))
        if not ((self.cmk_vault_base_url and self.cmk_key_name and self.cmk_key_version)
                or (not self.cmk_vault_base_url and not self.cmk_key_name and not self.cmk_key_version)):
            raise WebserviceException('customer managed key vault_base_url, key_name and key_version must all have values or \
                                      all are empty/null',
                                      logger=module_logger)
        if (self.vnet_name and not self.subnet_name) or (not self.vnet_name and self.subnet_name):
            raise WebserviceException('vnet_name and subnet_name must all have values or both are empty/null',
                                      logger=module_logger)

    def print_deploy_configuration(self):
        """Print the deployment configuration."""
        deploy_config = []
        if self.cpu_cores:
            deploy_config.append('CPU requirement: {}'.format(self.cpu_cores))
        if self.memory_gb:
            deploy_config.append('Memory requirement: {}GB'.format(self.memory_gb))

        if len(deploy_config) > 0:
            print(', '.join(deploy_config))

    def _build_create_payload(self, name, environment_image_request, overwrite=False):
        import copy
        from azureml._model_management._util import aci_specific_service_create_payload_template
        json_payload = copy.deepcopy(aci_specific_service_create_payload_template)
        base_payload = super(AciServiceDeploymentConfiguration,
                             self)._build_base_create_payload(name, environment_image_request)

        json_payload['containerResourceRequirements']['cpu'] = self.cpu_cores
        json_payload['containerResourceRequirements']['memoryInGB'] = self.memory_gb
        if self.auth_enabled is not None:
            json_payload['authEnabled'] = self.auth_enabled
        else:
            del(json_payload['authEnabled'])
        if self.enable_app_insights is not None:
            json_payload['appInsightsEnabled'] = self.enable_app_insights
        else:
            del(json_payload['appInsightsEnabled'])
        if self.collect_model_data:
            json_payload['dataCollection']['storageEnabled'] = self.collect_model_data
        else:
            del(json_payload['dataCollection'])

        if self.ssl_enabled is not None:
            json_payload['sslEnabled'] = self.ssl_enabled
            if self.ssl_enabled:
                try:
                    with open(self.ssl_cert_pem_file, 'r') as cert_file:
                        cert_data = cert_file.read()
                    json_payload['sslCertificate'] = cert_data
                except Exception as e:
                    raise WebserviceException('Error occurred attempting to read SSL cert pem file:\n{}'.format(e))
                try:
                    with open(self.ssl_key_pem_file, 'r') as key_file:
                        key_data = key_file.read()
                    json_payload['sslKey'] = key_data
                except Exception as e:
                    raise WebserviceException('Error occurred attempting to read SSL key pem file:\n{}'.format(e))
        else:
            del(json_payload['sslEnabled'])
            del(json_payload['sslCertificate'])
            del(json_payload['sslKey'])

        json_payload['cname'] = self.ssl_cname
        json_payload['dnsNameLabel'] = self.dns_name_label

        encryption_properties = {}
        vnet_configuration = {}

        # All 3 propertis will either be all null or all valid
        # validation is done in aciDeploymentConfiguration
        if self.cmk_vault_base_url:
            encryption_properties['vaultBaseUrl'] = self.cmk_vault_base_url
            encryption_properties['keyName'] = self.cmk_key_name
            encryption_properties['keyVersion'] = self.cmk_key_version
            json_payload['encryptionProperties'] = encryption_properties

        if self.vnet_name:
            vnet_configuration['vnetName'] = self.vnet_name
            vnet_configuration['subnetName'] = self.subnet_name
            json_payload['vnetConfiguration'] = vnet_configuration

        if overwrite:
            json_payload['overwrite'] = overwrite
        else:
            del (json_payload['overwrite'])

        json_payload.update(base_payload)

        return json_payload
