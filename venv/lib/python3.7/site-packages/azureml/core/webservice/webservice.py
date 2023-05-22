# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing models deployed as a web service endpoint in Azure Machine Learning.

This module contains the abstract parent class :class:`azureml.core.webservice.webservice.Webservice`,
which defines methods for deploying models. A common pattern is to create a configuration object for the
specific compute target, and then use the methods of the Webservice class with that configuration object.
For example, to deploy to Azure Container Instances, create a
:class:`azureml.core.webservice.aci.AciServiceDeploymentConfiguration` object from the ``deploy_configuration``
method of the :class:`azureml.core.webservice.aci.AciWebservice` class, and then use one of the deploy methods
of the Webservice class. A similar pattern applies for the :class:`azureml.core.webservice.AksWebservice`,
:class:`azureml.core.webservice.AksEndpoint`, and :class:`azureml.core.webservice.LocalWebservice` classes.

For an overview of deployment, see [Deploy models with Azure Machine
Learning](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-and-where).
"""

try:
    from abc import ABCMeta

    ABC = ABCMeta('ABC', (), {})
except ImportError:
    from abc import ABC
from abc import abstractmethod
import copy
import json
import logging
import os
import requests
import sys
import time
import warnings
from datetime import datetime
from dateutil import tz
from dateutil.parser import parse
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.exceptions import WebserviceException
from azureml._base_sdk_common.tracking import global_tracking_info_registry
from azureml._model_management._constants import MMS_SYNC_TIMEOUT_SECONDS
from azureml._model_management._constants import MMS_SERVICE_VALIDATE_OPERATION_TIMEOUT_SECONDS
from azureml._model_management._constants import MMS_SERVICE_EXIST_CHECK_SYNC_TIMEOUT_SECONDS
from azureml._model_management._constants import MMS_RESOURCE_CHECK_SYNC_TIMEOUT_SECONDS
from azureml._model_management._constants import CLOUD_DEPLOYABLE_IMAGE_FLAVORS
from azureml._model_management._constants import ACI_WEBSERVICE_TYPE, AKS_WEBSERVICE_TYPE, AKS_ENDPOINT_TYPE
from azureml._model_management._constants import ALL_WEBSERVICE_TYPES
from azureml._model_management._constants import LOCAL_WEBSERVICE_TYPE, UNKNOWN_WEBSERVICE_TYPE
from azureml._model_management._constants import SERVICE_REQUEST_OPERATION_CREATE, SERVICE_REQUEST_OPERATION_UPDATE
from azureml._model_management._constants import SERVICE_REQUEST_OPERATION_DELETE
from azureml._model_management._util import get_paginated_results
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import webservice_name_validation
from azureml._model_management._util import _get_mms_url
from azureml._restclient.clientbase import ClientBase

module_logger = logging.getLogger(__name__)


class Webservice(ABC):
    """
    Defines base functionality for deploying models as web service endpoints in Azure Machine Learning.

    Webservice constructor is used to retrieve a cloud representation of a Webservice object associated with the
    provided Workspace. Returns an instance of a child class corresponding to the specific type of the retrieved
    Webservice object. The Webservice class allows for deploying machine learning models from either a
    :class:`azureml.core.Model` or :class:`azureml.core.Image` object.

    For more information about working with Webservice, see `Deploy models
    with Azure Machine Learning <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-and-where>`_.

    .. remarks::

        The following sample shows the recommended deployment pattern where you first create a configuration object
        with the ``deploy_configuration`` method of the child class of Webservice (in this case
        :class:`azureml.core.webservice.AksWebservice`) and then use the configuration with the ``deploy`` method of
        the :class:`azureml.core.model.Model` class.

        .. code-block:: python

            # Set the web service configuration (using default here)
            aks_config = AksWebservice.deploy_configuration()

            # # Enable token auth and disable (key) auth on the webservice
            # aks_config = AksWebservice.deploy_configuration(token_auth_enabled=True, auth_enabled=False)

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/production-deploy-to-aks/production-deploy-to-aks.ipynb


        The following sample shows how to find an existing :class:`azureml.core.webservice.AciWebservice` in a
        workspace and delete it if it exists so the name can be reused.

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


        There are a number of ways to deploy a model as a webservice, including with the:

        * ``deploy`` method of the :class:`azureml.core.model.Model` for models already registered in the workspace.

        * ``deploy_from_image`` method of :class:`azureml.core.webservice.Webservice` for images already created from
          a model.

        * ``deploy_from_model`` method of :class:`azureml.core.webservice.Webservice` for models already registered
          in the workspace. This method will create an image.

        * ``deploy`` method of the :class:`azureml.core.webservice.Webservice`, which will register a model and
          create an image.

        For information on working with webservices, see

        * `Consume an Azure Machine Learning model deployed
          as a web service <https://docs.microsoft.com/azure/machine-learning/how-to-consume-web-service>`_

        * `Monitor and collect data from ML web service
          endpoints <https://docs.microsoft.com/azure/machine-learning/how-to-enable-app-insights>`_

        The *Variables* section lists attributes of a local representation of the cloud Webservice object. These
        variables should be considered read-only. Changing their values will not be reflected in the corresponding
        cloud object.

    :var auth_enabled: Whether or not the Webservice has auth enabled.
    :vartype auth_enabled: bool
    :var compute_type: What type of compute the Webservice is deployed to.
    :vartype compute_type: str
    :var created_time: When the Webservice was created.
    :vartype created_time: datetime.datetime
    :var azureml.core.Webservice.description: A description of the Webservice object.
    :vartype description: str
    :var azureml.core.Webservice.tags: A dictionary of tags for the Webservice object.
    :vartype tags: dict[str, str]
    :var azureml.core.Webservice.name: The name of the Webservice.
    :vartype name: str
    :var azureml.core.Webservice.properties: Dictionary of key value properties for the Webservice. These properties
        cannot be changed after deployment, however new key value pairs can be added.
    :vartype properties: dict[str, str]
    :var created_by: The user that created the Webservice.
    :vartype created_by: str
    :var error: If the Webservice failed to deploy, this will contain the error message for why it failed.
    :vartype error: str
    :var azureml.core.Webservice.state: The current state of the Webservice.
    :vartype state: str
    :var updated_time: The last time the Webservice was updated.
    :vartype updated_time: datetime.datetime
    :var azureml.core.Webservice.workspace: The Azure Machine Learning Workspace which contains this Webservice.
    :vartype workspace: azureml.core.Workspace
    :var token_auth_enabled: Whether or not the Webservice has token auth enabled.
    :vartype token_auth_enabled: bool

    :param workspace: The workspace object containing the Webservice object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the of the Webservice object to retrieve.
    :type name: str
    """

    # TODO: Add "createdBy" back to expected payload keys upon rollout of
    # https://msdata.visualstudio.com/Vienna/_git/model-management/pullrequest/290466?_a=overview
    _expected_payload_keys = ['computeType', 'createdTime', 'description', 'kvTags', 'name', 'properties']
    _webservice_type = None

    def __new__(cls, workspace, name):
        """Webservice constructor.

        The Webservice constructor retrieves a cloud representation of a Webservice object associated with the
        provided workspace. It will return an instance of a child class corresponding to the specific type of the
        retrieved Webservice object. A :class:`azureml.exceptions.WebserviceException` is raised if the Webservice
        isn't found.

        :param workspace: The workspace object containing the Webservice object to retrieve.
        :type workspace: azureml.core.Workspace
        :param name: The name of the of the Webservice object to retrieve.
        :type name: str
        :return: An instance of a child of Webservice corresponding to the specific type of the retrieved
            Webservice object.
        :rtype: azureml.core.Webservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if workspace and name:
            service_payload = cls._get(workspace, name)
            if service_payload:
                service_type = service_payload['computeType']
                child_class = None
                for child in Webservice._all_subclasses(Webservice):
                    if service_type == child._webservice_type:
                        child_class = child
                        break
                    elif child._webservice_type == UNKNOWN_WEBSERVICE_TYPE:
                        child_class = child

                if child_class:
                    service = super(Webservice, cls).__new__(child_class)
                    service._initialize(workspace, service_payload)
                    return service
            else:
                raise WebserviceException('WebserviceNotFound: Webservice with name {} not found in provided '
                                          'workspace'.format(name))
        else:
            return super(Webservice, cls).__new__(cls)

    @staticmethod
    def _all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in Webservice._all_subclasses(c)])

    def __init__(self, workspace, name):
        """Initialize the Webservice instance.

        The Webservice constructor retrieves a cloud representation of a Webservice object associated with the
        provided workspace. It will return an instance of a child class corresponding to the specific type of the
        retrieved Webservice object.

        :param workspace: The workspace object containing the Webservice object to retrieve.
        :type workspace: azureml.core.Workspace
        :param name: The name of the of the Webservice object to retrieve.
        :type name: str
        :return: An instance of a child of Webservice corresponding to the specific type of the retrieved
            Webservice object.
        :rtype: azureml.core.Webservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        pass

    def __repr__(self):
        """Return the string representation of the Webservice object.

        :return: String representation of the Webservice object
        :rtype: str
        """
        return "{}(workspace={}, name={}, image_id={}, image_digest={}, compute_type={}, state={}, scoring_uri={}, " \
            "tags={}, properties={}, created_by={})".format(
                self.__class__.__name__,
                self.workspace.__repr__() if hasattr(self, 'workspace') else None,
                self.name if hasattr(self, 'name') else None,
                self.image_id if hasattr(self, 'image_id') else None,
                self.image_digest if hasattr(self, 'image_digest') else None,
                self.compute_type if hasattr(self, 'compute_type') else None,
                self.state if hasattr(self, 'state') else None,
                self.scoring_uri if hasattr(self, 'scoring_uri') else None,
                self.tags if hasattr(self, 'tags') else None,
                self.properties if hasattr(self, 'properties') else None,
                self.created_by if hasattr(self, 'created_by') else None)

    @property
    def _webservice_session(self):
        self._session = requests.Session()
        self._session.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json'})

        if self.auth_enabled:
            service_keys = self.get_keys()
            self._session.headers.update({'Authorization': 'Bearer ' + service_keys[0]})
        if self.token_auth_enabled:
            service_token, self._refresh_token_time = self.get_token()
            self._session.headers.update({'Authorization': 'Bearer ' + service_token})

        if self._refresh_token_time and self._refresh_token_time < datetime.utcnow():
            try:
                service_token, self._refresh_token_time = self.get_token()
                self._session.headers.update({'Authorization': 'Bearer ' + service_token})
            except WebserviceException:
                pass  # Tokens are valid for 12 hours pass the refresh time so if we can't refresh it now, try later

        return self._session

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
        # Expected payload keys
        self.auth_enabled = obj_dict.get('authEnabled')
        self.compute_type = obj_dict.get('computeType')
        self.created_time = parse(obj_dict.get('createdTime'))
        self.description = obj_dict.get('description')
        self.image_id = obj_dict.get('imageId')
        self.image_digest = obj_dict.get('imageDigest')
        self.tags = obj_dict.get('kvTags')
        self.name = obj_dict.get('name')
        self.properties = obj_dict.get('properties')
        self.created_by = obj_dict.get('createdBy')

        # Common amongst Webservice classes but optional payload keys
        self.error = obj_dict.get('error')
        self.image = Image.deserialize(workspace, obj_dict['imageDetails']) if 'imageDetails' in obj_dict else None
        self.state = obj_dict.get('state')
        self.updated_time = parse(obj_dict['updatedTime']) if 'updatedTime' in obj_dict else None

        # Utility payload keys
        self._auth = workspace._auth_object
        self._operation_endpoint = None
        self._mms_endpoint = _get_mms_url(workspace) + '/services/{}'.format(self.name)
        self.workspace = workspace
        self._session = None

        self.token_auth_enabled = False  # Can be overridden in other webservices (currently only AKS).
        self._refresh_token_time = None

    @staticmethod
    def _get(workspace, name=None):
        """Get the Webservice with the corresponding name.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :return:
        :rtype: dict
        """
        if not name:
            raise WebserviceException('Name must be provided', logger=module_logger)

        base_url = _get_mms_url(workspace)
        mms_url = base_url + '/services'

        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth_object.get_authentication_header())
        params = {'expand': 'true'}

        service_url = mms_url + '/{}'.format(name)

        resp = ClientBase._execute_func(get_requests_session().get, service_url, headers=headers, params=params)
        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            service_payload = json.loads(content)
            return service_payload
        elif resp.status_code == 404:
            return None
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    @staticmethod
    def deploy(workspace, name, model_paths, image_config, deployment_config=None, deployment_target=None,
               overwrite=False):  # pragma: no cover
        """
        Deploy a Webservice from zero or more :class:`azureml.core.Model` objects.

        This function will register any models files provided and create an image in the process,
        all associated with the specified :class:`azureml.core.Workspace`. Use this function when you have a
        directory of models to deploy that haven't been previously registered.

        The resulting Webservice is a real-time endpoint that can be used for inference requests.
        For more information, see `Consume a model deployed as a web
        service <https://docs.microsoft.com/azure/machine-learning/how-to-consume-web-service>`_.

        :param workspace: A Workspace object to associate the Webservice with.
        :type workspace: azureml.core.Workspace
        :param name: The name to give the deployed service. Must be unique to the workspace, only consist of lowercase
            letters, numbers, or dashes, start with a letter, and be between 3 and 32 characters long.
        :type name: str
        :param model_paths: A list of on-disk paths to model files or folder. Can be an empty list.
        :type model_paths: builtin.list[str]
        :param image_config: An ImageConfig object used to determine required Image properties.
        :type image_config: azureml.core.image.image.ImageConfig
        :param deployment_config: A WebserviceDeploymentConfiguration used to configure the webservice. If one is not
            provided, an empty configuration object will be used based on the desired target.
        :type deployment_config: azureml.core.webservice.webservice.WebserviceDeploymentConfiguration
        :param deployment_target: A :class:`azureml.core.ComputeTarget` to deploy the Webservice to. As Azure
            Container Instances has no associated :class:`azureml.core.ComputeTarget`, leave this parameter
            as None to deploy to Azure Container Instances.
        :type deployment_target: azureml.core.ComputeTarget
        :param overwrite: Overwrite the existing service if service with name already exists.
        :type overwrite: bool
        :return: A Webservice object corresponding to the deployed webservice.
        :rtype: azureml.core.Webservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        webservice_name_validation(name)
        Webservice._check_for_local_deployment(deployment_config)
        models = []
        for model_path in model_paths:
            model_name = os.path.basename(model_path.rstrip(os.sep))[:30]
            models.append(Model.register(workspace, model_path, model_name))
        return Webservice.deploy_from_model(workspace, name, models, image_config, deployment_config,
                                            deployment_target, overwrite)

    @staticmethod
    def deploy_from_model(workspace, name, models, image_config, deployment_config=None, deployment_target=None,
                          overwrite=False):  # pragma: no cover
        """
        Deploy a Webservice from zero or more :class:`azureml.core.Model` objects.

        This function is similar to :func:`deploy`, but does not register the models. Use this function if you have
        model objects that are already registered. This will create an image in the process, associated with the
        specified Workspace.

        The resulting Webservice is a real-time endpoint that can be used for inference requests.
        For more information, see `Consume a model deployed as a web
        service <https://docs.microsoft.com/azure/machine-learning/how-to-consume-web-service>`_.

        :param workspace: A Workspace object to associate the Webservice with.
        :type workspace: azureml.core.Workspace
        :param name: The name to give the deployed service. Must be unique to the workspace, only consist of lowercase
            letters, numbers, or dashes, start with a letter, and be between 3 and 32 characters long.
        :type name: str
        :param models: A list of model objects. Can be an empty list.
        :type models: builtin.list[azureml.core.Model]
        :param image_config: An ImageConfig object used to determine required Image properties.
        :type image_config: azureml.core.image.image.ImageConfig
        :param deployment_config: A WebserviceDeploymentConfiguration used to configure the webservice. If one is not
            provided, an empty configuration object will be used based on the desired target.
        :type deployment_config: azureml.core.webservice.webservice.WebserviceDeploymentConfiguration
        :param deployment_target: A :class:`azureml.core.ComputeTarget` to deploy the Webservice to. As ACI has no
            associated :class:`azureml.core.ComputeTarget`, leave this parameter as None to deploy to ACI.
        :type deployment_target: azureml.core.ComputeTarget
        :param overwrite: Overwrite the existing service if service with name already exists.
        :type overwrite: bool
        :return: A Webservice object corresponding to the deployed webservice.
        :rtype: azureml.core.Webservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        warnings.warn("deploy_from_model has been deprecated and will be removed in a future release. "
                      + "Please migrate to using Environments. "
                      + "https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments",
                      category=DeprecationWarning, stacklevel=2)

        webservice_name_validation(name)
        Webservice._check_for_local_deployment(deployment_config)

        image = Image.create(workspace, name, models, image_config)
        image.wait_for_creation(True)
        if image.creation_state != 'Succeeded':
            raise WebserviceException('Error occurred creating image {} for service. More information can be found '
                                      'here: {}\n Generated DockerFile can be found here: {}'.format(
                                          image.id,
                                          image.image_build_log_uri,
                                          image.generated_dockerfile_uri), logger=module_logger)
        return Webservice.deploy_from_image(workspace, name, image, deployment_config, deployment_target, overwrite)

    @staticmethod
    def deploy_from_image(workspace, name, image, deployment_config=None, deployment_target=None,
                          overwrite=False):  # pragma: no cover
        """
        Deploy a Webservice from an :class:`azureml.core.Image` object.

        Use this function if you already have an Image object created for a model.

        The resulting Webservice is a real-time endpoint that can be used for inference requests.
        For more information, see `Consume a model deployed as a web
        service <https://docs.microsoft.com/azure/machine-learning/how-to-consume-web-service>`_.

        :param workspace: A Workspace object to associate the Webservice with.
        :type workspace: azureml.core.Workspace
        :param name: The name to give the deployed service. Must be unique to the workspace, only consist of lowercase
            letters, numbers, or dashes, start with a letter, and be between 3 and 32 characters long.
        :type name: str
        :param image: An :class:`azureml.core.Image` object to deploy.
        :type image: azureml.core.Image
        :param deployment_config: A WebserviceDeploymentConfiguration used to configure the webservice. If one is not
            provided, an empty configuration object will be used based on the desired target.
        :type deployment_config: azureml.core.webservice.webservice.WebserviceDeploymentConfiguration
        :param deployment_target: A :class:`azureml.core.ComputeTarget` to deploy the Webservice to. As Azure
            Container Instances has no associated :class:`azureml.core.ComputeTarget`, leave this parameter as
            None to deploy to Azure Container Instances.
        :type deployment_target: azureml.core.ComputeTarget
        :param overwrite: Overwrite the existing service if service with name already exists.
        :type overwrite: bool
        :return: A Webservice object corresponding to the deployed webservice.
        :rtype: azureml.core.Webservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        warnings.warn("deploy_from_image has been deprecated and will be removed in a future release. "
                      + "Please migrate to using Environments. "
                      + "https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments",
                      category=DeprecationWarning, stacklevel=2)

        webservice_name_validation(name)
        Webservice._check_for_local_deployment(deployment_config)
        if deployment_target is None:
            if deployment_config is None:
                for child in Webservice.__subclasses__():  # This is a hack to avoid recursive imports
                    if child._webservice_type == ACI_WEBSERVICE_TYPE:
                        return child._deploy(workspace, name, image, deployment_config, overwrite)
            return deployment_config._webservice_type._deploy(workspace, name, image, deployment_config,
                                                              overwrite=overwrite)

        else:
            if deployment_config is None:
                for child in Webservice.__subclasses__():  # This is a hack to avoid recursive imports
                    if child._webservice_type == AKS_WEBSERVICE_TYPE:
                        return child._deploy(workspace, name, image, deployment_config, deployment_target, overwrite)

        return deployment_config._webservice_type._deploy(workspace, name, image, deployment_config, deployment_target,
                                                          overwrite)

    @staticmethod
    def _check_validate_error(content):
        payload = json.loads(content)

        if payload and "error" in payload and "message" in payload["error"]:
            return payload["error"]["message"]
        else:
            return None

    @staticmethod
    def _generate_common_validation_payload(name, payload, action):
        common_payload = {'type': action,
                          'serviceName': name,
                          'createRequest': None,
                          'updateRequest': None}

        if action == SERVICE_REQUEST_OPERATION_CREATE:
            common_payload["createRequest"] = payload
            return common_payload
        elif action == SERVICE_REQUEST_OPERATION_UPDATE:
            common_payload["updateRequest"] = payload
            return common_payload
        elif action == SERVICE_REQUEST_OPERATION_DELETE:
            return common_payload
        else:
            return None

    @staticmethod
    def _request_validate_service_name(workspace, name):
        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth_object.get_authentication_header())
        params = {}
        body = {}
        base_url = _get_mms_url(workspace)
        mms_endpoint = base_url + '/services/validate/name/{}'.format(name)
        try:
            resp = ClientBase._execute_func(get_requests_session().post, mms_endpoint,
                                            params=params,
                                            timeout=MMS_SERVICE_EXIST_CHECK_SYNC_TIMEOUT_SECONDS,
                                            headers=headers,
                                            json=body)
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            return None

        if resp.status_code != 200:
            return None

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        return content

    @staticmethod
    def _request_validate_resource_not_enough(workspace, payload):
        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth_object.get_authentication_header())
        params = {}
        base_url = _get_mms_url(workspace)
        mms_endpoint = base_url + '/services/validate/resource'
        try:
            resp = ClientBase._execute_func(get_requests_session().post, mms_endpoint,
                                            params=params,
                                            timeout=MMS_RESOURCE_CHECK_SYNC_TIMEOUT_SECONDS,
                                            headers=headers,
                                            json=payload)
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            return None

        if resp.status_code != 200:
            return None

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        return content

    @staticmethod
    def _request_validate_service(workspace, name, payload, action):
        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth_object.get_authentication_header())
        params = {}
        base_url = _get_mms_url(workspace)
        mms_endpoint = base_url + '/services/validate'
        try:
            resp = ClientBase._execute_func(get_requests_session().post, mms_endpoint,
                                            params=params,
                                            timeout=MMS_SERVICE_VALIDATE_OPERATION_TIMEOUT_SECONDS,
                                            headers=headers,
                                            json=Webservice._generate_common_validation_payload(name, payload, action))
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            return None

        if resp.status_code != 200:
            return None

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        return content

    @staticmethod
    def _run_validate_framework(request_func, check_func):
        # a general validate framework for internal usage

        # request and get response content
        content = request_func()
        if not content:
            return

        # get message from deserialized data
        error = check_func(content)

        if not error:
            return

        raise WebserviceException(error)

    @staticmethod
    def check_for_existing_webservice(workspace, name, overwrite=False, request_func=None, check_func=None):
        """Check webservice exists.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param overwrite:
        :type overwrite: bool
        :param request_func: function to request service to check if service name exists
        :type request_func: function
        :param check_func: function to check response content of request_func
        :type check_func: function
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if overwrite:
            return

        if not request_func:
            def request_func():
                return Webservice._request_validate_service_name(workspace, name)

        if not check_func:
            def check_func(content):
                return Webservice._check_validate_error(content)

        Webservice._run_validate_framework(request_func, check_func)

    @staticmethod
    def _get_deploy_compute_type(deploy_payload):
        if deploy_payload \
                and 'computeType' in deploy_payload:
            return deploy_payload['computeType']

        return None

    @staticmethod
    def _support_validation_check(compute_type):
        if compute_type in (AKS_WEBSERVICE_TYPE, AKS_ENDPOINT_TYPE, ACI_WEBSERVICE_TYPE):
            # Currently only validate Aks/AksEndpoint/Aci.
            return True

        return False

    @staticmethod
    def _check_for_compute_resource(workspace, payload, request_func=None, check_func=None):
        """Check Deployment Compute Resource Request.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param payload:
        :type payload: dict
        :param request_func: function to request service to check if the compute resource is sufficient
        :type request_func: function
        :param check_func: function to check response content of request_func
        :type check_func: function
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if not Webservice._support_validation_check(Webservice._get_deploy_compute_type(payload)):
            # TODO Support update action for resource check, currently resource validation only supports deploy action.
            return

        if not request_func:
            def request_func():
                return Webservice._request_validate_resource_not_enough(workspace, payload)

        if not check_func:
            def check_func(content):
                return Webservice._check_validate_error(content)

        Webservice._run_validate_framework(request_func, check_func)

    @staticmethod
    def _check_for_webservice(workspace, name, compute_type, payload, action, request_func=None, check_func=None):
        """Webservice General Check.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param compute_type:
        :type compute_type: str
        :param payload:
        :type payload: Any
        :param action:
        :type action: str
        :param request_func: function to request the general check of the webservice deployment.
        :type request_func: function
        :param check_func: function to check response content of request_func
        :type check_func: function
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if not Webservice._support_validation_check(compute_type):
            return

        if not request_func:
            def request_func():
                return Webservice._request_validate_service(workspace, name, payload, action)

        if not check_func:
            def check_func(content):
                return Webservice._check_validate_error(content)

        Webservice._run_validate_framework(request_func, check_func)

    @staticmethod
    def _check_for_local_deployment(deployment_config):  # pragma: no cover
        from azureml.core.webservice.local import LocalWebserviceDeploymentConfiguration
        if deployment_config and (type(deployment_config) is LocalWebserviceDeploymentConfiguration):
            raise WebserviceException('This method does not support local deployment configuration. Please use '
                                      'deploy_local_from_model for local deployment.')

    @staticmethod
    def deploy_local_from_model(workspace, name, models, image_config, deployment_config=None,
                                wait=False):  # pragma: no cover
        """
        Build and deploy a :class:`azureml.core.webservice.local.LocalWebservice` for testing.

        Requires Docker to be installed and configured.

        :param workspace: A Workspace object with which to associate the Webservice.
        :type workspace: azureml.core.Workspace
        :param name: The name to give the deployed service. Must be unique on the local machine.
        :type name: str
        :param models: A list of model objects. Can be an empty list.
        :type models: builtin.list[azureml.core.Model]
        :param image_config: An ImageConfig object used to determine required service image properties.
        :type image_config: azureml.core.image.image.ImageConfig
        :param deployment_config: A LocalWebserviceDeploymentConfiguration used to configure the webservice. If one is
            not provided, an empty configuration object will be used.
        :type deployment_config: azureml.core.webservice.local.LocalWebserviceDeploymentConfiguration
        :param wait: Whether to wait for the LocalWebservice's Docker container to report as healthy.
            Throws an exception if the container crashes. The default is False.
        :type wait: bool
        :rtype: azureml.core.webservice.LocalWebservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        webservice_name_validation(name)

        for child in Webservice.__subclasses__():  # This is a hack to avoid recursive imports
            if child._webservice_type == LOCAL_WEBSERVICE_TYPE:
                return child._deploy(workspace, name, models, image_config, deployment_config, wait)

    @staticmethod
    @abstractmethod
    def _deploy(workspace, name, image, deployment_config, deployment_target, overwrite=False):
        """Deploy the Webservice to the cloud.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param image:
        :type image: azureml.core.Image
        :param deployment_config:
        :type deployment_config: WebserviceDeploymentConfiguration
        :param deployment_target:
        :type deployment_target: azureml.core.ComputeTarget
        :param overwrite:
        :type overwrite: bool
        :return:
        :rtype: azureml.core.Webservice
        """
        pass

    @staticmethod
    def _deploy_webservice(workspace, name, webservice_payload, overwrite, webservice_class, show_output=False):
        """Deploy the Webservice to the cloud.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param webservice_payload:
        :type webservice_payload: dict
        :param overwrite:
        :type overwrite: bool
        :param webservice_class:
        :type webservice_class: azureml.core.Webservice
        :param show_output: Indicates whether to display the configuration of service deployment.
        :type show_output: bool
        :return:
        :rtype: azureml.core.Webservice
        """
        # TODO Remove check_for_existing_webservice() later, this check has already included in common validation.
        Webservice.check_for_existing_webservice(workspace, name, overwrite)
        Webservice._check_for_webservice(workspace, name, Webservice._get_deploy_compute_type(webservice_payload),
                                         webservice_payload, SERVICE_REQUEST_OPERATION_CREATE)

        if not overwrite:
            Webservice._check_for_compute_resource(workspace, webservice_payload)

        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth_object.get_authentication_header())
        params = {}
        base_url = _get_mms_url(workspace)
        mms_endpoint = base_url + '/services'

        try:
            resp = ClientBase._execute_func(get_requests_session().post, mms_endpoint, params=params, headers=headers,
                                            json=webservice_payload)
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        if resp.status_code >= 400:
            raise WebserviceException('Error occurred creating service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        if 'Operation-Location' in resp.headers:
            operation_location = resp.headers['Operation-Location']
        else:
            raise WebserviceException('Missing response header key: Operation-Location', resp.status_code,
                                      logger=module_logger)
        create_operation_status_id = operation_location.split('/')[-1]
        operation_url = base_url + '/operations/{}'.format(create_operation_status_id)

        service = webservice_class(workspace, name=name)
        service._operation_endpoint = operation_url

        if show_output:
            print('Request submitted, please run wait_for_deployment(show_output=True) to get deployment status.')

        return service

    def wait_for_deployment(self, show_output=False, timeout_sec=None):
        """Automatically poll on the running Webservice deployment.

        Wait for the Webservice to reach a terminal state. Will throw a
        :class:`azureml.exceptions.WebserviceException` if it reaches a non-successful terminal state
        or exceeds the provided timeout.

        :param show_output: Indicates whether to print more verbose output.
        :type show_output: bool
        :param timeout_sec: Raise an exception if deployment exceeds the given timeout.
        :type timeout_sec: float
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if show_output:
            print('Tips: You can try get_logs(): https://aka.ms/debugimage#dockerlog '
                  'or local deployment: https://aka.ms/debugimage#debug-locally to debug '
                  'if deployment takes longer than 10 minutes.')

        if timeout_sec:
            timeout_sec = float(timeout_sec)

        try:
            operation_state, error, operation = self._wait_for_operation_to_complete(show_output, timeout_sec)
            self.update_deployment_state()
            if operation_state != 'Succeeded':
                if error:  # Operation response error
                    error_response = json.dumps(error, indent=2)
                elif self.error:  # Service response error
                    error_response = json.dumps(self.error, indent=2)
                else:
                    error_response = 'No error message received from server.'

                format_error_response = Webservice._format_error_response(error_response)
                logs_response = None
                operation_details = operation.get('operationDetails')
                if operation_details:
                    sub_op_type = operation_details.get('subOperationType')
                    if sub_op_type:
                        if sub_op_type == 'BuildEnvironment' or sub_op_type == 'BuildImage':
                            operation_log_uri = operation.get('operationLog')
                            logs_response = 'More information can be found here: {}'.format(operation_log_uri)
                        elif sub_op_type == 'DeployService':
                            logs_response = 'More information can be found using \'.get_logs()\''
                if not logs_response:
                    logs_response = 'Current sub-operation type not known, more logs unavailable.'

                raise WebserviceException('Service deployment polling reached non-successful terminal state, current '
                                          'service state: {}\n'
                                          'Operation ID: {}\n'
                                          '{}\n'
                                          'Error:\n'
                                          '{}'.format(self.state, self._operation_endpoint.split('/')[-1],
                                                      logs_response, format_error_response), logger=module_logger)
            print('{} service creation operation finished, operation "{}"'.format(self._webservice_type,
                                                                                  operation_state))
        except WebserviceException as e:
            if e.message == 'No operation endpoint':
                self.update_deployment_state()
                raise WebserviceException('Long running operation information not known, unable to poll. '
                                          'Current state is {}'.format(self.state), logger=module_logger)
            else:
                raise

    def _wait_for_operation_to_complete(self, show_output, timeout_sec):
        """Poll on the async operation for this Webservice.

        :param show_output:
        :type show_output: bool
        :return:
        :rtype: (str, str, dict)
        """
        if not self._operation_endpoint:
            raise WebserviceException('No operation endpoint', logger=module_logger)
        state, error, operation = self._get_operation_state()
        current_state = state
        start_time, elapsed_time = time.time(), 0.0
        if show_output:
            sys.stdout.write('{}'.format(current_state))
            sys.stdout.flush()

        streaming_log_offset = 0
        while state != 'Succeeded' and state != 'Failed' and state != 'Cancelled' and state != 'TimedOut':
            if timeout_sec and elapsed_time >= timeout_sec:
                msg = 'Deployment polling exceeded timeout override of {}s. Current status: {}\nDeployment may be \
                    transitioning or unhealthy, use update_deployment_state() to fetch the latest status of \
                        the webservice'.format(timeout_sec, current_state)
                raise WebserviceException(msg, logger=module_logger)

            time.sleep(5)

            elapsed_time = time.time() - start_time
            state, error, operation = self._get_operation_state(show_output, streaming_log_offset)
            if show_output:
                if 'streamingOperationLog' not in operation:
                    sys.stdout.write('.')
                else:
                    streaming_log = operation['streamingOperationLog']
                    if streaming_log:
                        streaming_log_offset += len(streaming_log)
                        streaming_logs = streaming_log.split('#')
                        for log in streaming_logs:
                            time_and_log = log.split('|')
                            if len(time_and_log) != 2:
                                sys.stdout.write('.')
                            else:
                                utc_time = time_and_log[0]
                                utc_time = datetime.strptime(utc_time, '%Y%m%d%H%M%S').replace(tzinfo=tz.tzutc())
                                local_time = utc_time.astimezone(tz.tzlocal())
                                sys.stdout.write('\n' + str(local_time) + ' ' + time_and_log[1])
                if state != current_state:
                    sys.stdout.write('\n{}\n'.format(state))
                    current_state = state
                sys.stdout.flush()

        return state, error, operation

    @staticmethod
    def _format_error_response(error_response):
        """Format mms returned error message to make it more readable.

        :param error_response: the mms returned error message str.
        :type error_response: str
        :return:
        :rtype: str
        """
        try:
            # error_response returned may have some 2-times escapes, so here need decode twice to un-escape all.
            format_error_response = error_response.encode('utf-8').decode('unicode_escape')
            format_error_response = format_error_response.encode('utf-8').decode('unicode_escape')
            return format_error_response
        except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
            return error_response

    def _get_operation_state(self, show_output=False, offset=0):
        """Get the current async operation state for the Webservice.

        :return:
        :rtype: str, str, dict
        """
        headers = self._auth.get_authentication_header()
        if show_output:
            params = {'offset': offset}
        else:
            params = {}

        resp = ClientBase._execute_func(get_requests_session().get, self._operation_endpoint, headers=headers,
                                        params=params, timeout=MMS_SYNC_TIMEOUT_SECONDS)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Resource Provider:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = json.loads(content)
        state = content['state']
        error = content.get('error')
        return state, error, content

    def update_deployment_state(self):
        """
        Refresh the current state of the in-memory object.

        Perform an in-place update of the properties of the object based on the current state of the corresponding
        cloud object. Primarily useful for manual polling of creation state.
        """
        service = Webservice(self.workspace, name=self.name)
        for key in self.__dict__.keys():
            if key != "_operation_endpoint":
                self.__dict__[key] = service.__dict__[key]

    @staticmethod
    def list(workspace, compute_type=None, image_name=None, image_id=None, model_name=None, model_id=None, tags=None,
             properties=None, image_digest=None):
        """
        List the Webservices associated with the corresponding :class:`azureml.core.Workspace`.

        The results returned can be filtered using parameters.

        :param workspace: The Workspace object to list the Webservices in.
        :type workspace: azureml.core.Workspace
        :param compute_type: Filter to list only specific Webservice types. Options are 'ACI', 'AKS'.
        :type compute_type: str
        :param image_name: Filter list to only include Webservices deployed with the specific image name.
        :type image_name: str
        :param image_id: Filter list to only include Webservices deployed with the specific image ID.
        :type image_id: str
        :param model_name: Filter list to only include Webservices deployed with the specific model name.
        :type model_name: str
        :param model_id: Filter list to only include Webservices deployed with the specific model ID.
        :type model_id: str
        :param tags: Filter based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type tags: builtin.list
        :param properties: Filter based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type properties: builtin.list
        :param image_digest: Filter list to only include Webservices deployed with the specific image digest.
        :type image_digest: str
        :return: A filtered list of Webservices in the provided Workspace.
        :rtype: builtin.list[azureml.core.Webservice]
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        webservices = []
        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth_object.get_authentication_header())
        params = {'expand': 'true'}

        base_url = _get_mms_url(workspace)
        mms_workspace_url = base_url + '/services'

        if compute_type:
            if compute_type.upper() not in ALL_WEBSERVICE_TYPES:
                raise WebserviceException('Invalid compute type "{}". Valid options are "{}"'
                                          .format(compute_type, ",".join(ALL_WEBSERVICE_TYPES)),
                                          logger=module_logger)
            params['computeType'] = compute_type
        if image_name:  # pragma: no cover
            params['imageName'] = image_name
        if image_id:  # pragma: no cover
            params['imageId'] = image_id
        if model_name:
            params['modelName'] = model_name
        if model_id:
            params['modelId'] = model_id
        if tags:
            tags_query = ""
            for tag in tags:
                if type(tag) is list:
                    tags_query = tags_query + tag[0] + "=" + tag[1] + ","
                else:
                    tags_query = tags_query + tag + ","
            tags_query = tags_query[:-1]
            params['tags'] = tags_query
        if properties:
            properties_query = ""
            for prop in properties:
                if type(prop) is list:
                    properties_query = properties_query + prop[0] + "=" + prop[1] + ","
                else:
                    properties_query = properties_query + prop + ","
            properties_query = properties_query[:-1]
            params['properties'] = properties_query
        if image_digest:
            params['imageDigest'] = image_digest

        try:
            resp = ClientBase._execute_func(get_requests_session().get, mms_workspace_url, headers=headers,
                                            params=params, timeout=MMS_SYNC_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.Timeout:
            raise WebserviceException('Error, request to MMS timed out to URL: {}'.format(mms_workspace_url),
                                      logger=module_logger)
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        services_payload = json.loads(content)
        paginated_results = get_paginated_results(services_payload, headers)
        for service_dict in paginated_results:
            service_type = service_dict['computeType']
            child_class = None

            for child in Webservice._all_subclasses(Webservice):
                if service_type == child._webservice_type:
                    child_class = child
                    break
                elif child._webservice_type == UNKNOWN_WEBSERVICE_TYPE:
                    child_class = child

            if child_class:
                service_obj = child_class.deserialize(workspace, service_dict)
                webservices.append(service_obj)
        return webservices

    def _add_tags(self, tags):
        """Add tags to this Webservice.

        :param tags:
        :type tags: dict[str, str]
        :return:
        :rtype: dict[str, str]
        """
        updated_tags = self.tags
        if updated_tags is None:
            return copy.deepcopy(tags)
        else:
            for key in tags:
                if key in updated_tags:
                    print("Replacing tag {} -> {} with {} -> {}".format(key, updated_tags[key], key, tags[key]))
                updated_tags[key] = tags[key]

        return updated_tags

    def _remove_tags(self, tags):
        """Remove tags from this Webservice.

        :param tags:
        :type tags: builtin.list[str]
        :return:
        :rtype: builtin.list[str]
        """
        updated_tags = self.tags
        if updated_tags is None:
            print('Model has no tags to remove.')
            return updated_tags
        else:
            if type(tags) is not list:
                tags = [tags]
            for key in tags:
                if key in updated_tags:
                    del updated_tags[key]
                else:
                    print('Tag with key {} not found.'.format(key))

        return updated_tags

    def _add_properties(self, properties):
        """Add properties to this Webservice.

        :param properties:
        :type properties: dict[str, str]
        :return:
        :rtype: dict[str, str]
        """
        updated_properties = self.properties
        if updated_properties is None:
            return copy.deepcopy(properties)
        else:
            for key in properties:
                if key in updated_properties:
                    print("Replacing tag {} -> {} with {} -> {}".format(key, updated_properties[key],
                                                                        key, properties[key]))
                updated_properties[key] = properties[key]

        return updated_properties

    def get_logs(self, num_lines=5000, init=False):
        """Retrieve logs for this Webservice.

        :param num_lines: The maximum number of log lines to retrieve.
        :type num_lines: int
        :param init: Get logs of init container
        :type init: bool
        :return: The logs for this Webservice.
        :rtype: str
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        headers = {'Content-Type': 'application/json'}
        headers.update(self.workspace._auth_object.get_authentication_header())
        params = {'tail': num_lines, 'init': init}
        service_logs_url = self._mms_endpoint + '/logs'

        resp = ClientBase._execute_func(get_requests_session().get, service_logs_url, headers=headers, params=params)
        if resp.status_code >= 400:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        else:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            service_payload = json.loads(content)
            if 'content' not in service_payload:
                raise WebserviceException('Invalid response, missing "content":\n'
                                          '{}'.format(service_payload), logger=module_logger)
            else:
                return service_payload['content']

    @abstractmethod
    def run(self, input):
        """
        Call this Webservice with the provided input.

        Abstract method implemented by child classes of :class:`azureml.core.Webservice`.

        :param input: The input data to call the Webservice with. This is the data your machine learning model expects
            as an input to run predictions.
        :type input: varies
        :return: The result of calling the Webservice. This will return predictions run from your machine
            learning model.
        :rtype: dict
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        pass

    def get_keys(self):
        """Retrieve auth keys for this Webservice.

        :return: The auth keys for this Webservice.
        :rtype: (str, str)
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        headers = self._auth.get_authentication_header()
        params = {}
        list_keys_url = self._mms_endpoint + '/listkeys'

        try:
            resp = ClientBase._execute_func(get_requests_session().post, list_keys_url, params=params, headers=headers)
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        keys_content = json.loads(content)
        if 'primaryKey' not in keys_content:
            raise WebserviceException('Invalid response key: primaryKey', logger=module_logger)
        if 'secondaryKey' not in keys_content:
            raise WebserviceException('Invalid response key: secondaryKey', logger=module_logger)
        primary_key = keys_content['primaryKey']
        secondary_key = keys_content['secondaryKey']

        return primary_key, secondary_key

    def regen_key(self, key, set_key=None):
        """Regenerate one of the Webservice's keys, either the 'Primary' or 'Secondary' key.

        A :class:`azureml.exceptions.WebserviceException` is raised if ``key`` is not specified or is not 'Primary'
        or 'Secondary'.

        :param key: The key to regenerate. Options are 'Primary' or 'Secondary'.
        :type key: str
        :param set_key: A user specified value allowing for manual specification of the key's value
        :type set_key: str
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        headers = {'Content-Type': 'application/json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        if not key:
            raise WebserviceException('Error, must specify which key with be regenerated: Primary, Secondary',
                                      logger=module_logger)
        key = key.capitalize()
        if key != 'Primary' and key != 'Secondary':
            raise WebserviceException('Error, invalid value provided for key: {}.\n'
                                      'Valid options are: Primary, Secondary'.format(key), logger=module_logger)
        keys_url = self._mms_endpoint + '/regenerateKeys'
        body = {'keyType': key,
                'keyValue': set_key}
        try:
            resp = ClientBase._execute_func(get_requests_session().post, keys_url, params=params, headers=headers,
                                            json=body)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise WebserviceException('Error connecting to {}.'.format(keys_url), logger=module_logger)
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        if 'Operation-Location' in resp.headers:
            operation_location = resp.headers['Operation-Location']
        else:
            raise WebserviceException('Missing operation location from response header, unable to determine status.',
                                      logger=module_logger)
        create_operation_status_id = operation_location.split('/')[-1]
        operation_endpoint = _get_mms_url(self.workspace) + '/operations/{}'.format(create_operation_status_id)
        operation_state = 'Running'
        while operation_state != 'Cancelled' and operation_state != 'Succeeded' and operation_state != 'Failed' \
                and operation_state != 'TimedOut':
            time.sleep(5)
            try:
                operation_resp = ClientBase._execute_func(get_requests_session().get, operation_endpoint,
                                                          params=params, headers=headers,
                                                          timeout=MMS_SYNC_TIMEOUT_SECONDS)
                operation_resp.raise_for_status()
            except requests.ConnectionError:
                raise WebserviceException('Error connecting to {}.'.format(operation_endpoint), logger=module_logger)
            except requests.Timeout:
                raise WebserviceException('Error, request to {} timed out.'.format(operation_endpoint),
                                          logger=module_logger)
            except requests.exceptions.HTTPError:
                raise WebserviceException('Received bad response from Model Management Service:\n'
                                          'Response Code: {}\n'
                                          'Headers: {}\n'
                                          'Content: {}'.format(operation_resp.status_code, operation_resp.headers,
                                                               operation_resp.content), logger=module_logger)
            content = operation_resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            content = json.loads(content)
            if 'state' in content:
                operation_state = content['state']
            else:
                raise WebserviceException('Missing state from operation response, unable to determine status',
                                          logger=module_logger)
            error = content['error'] if 'error' in content else None
        if operation_state != 'Succeeded':
            raise WebserviceException('Error, key regeneration operation "{}" with message '
                                      '"{}"'.format(operation_state, error), logger=module_logger)

    def get_token(self):
        """
        Retrieve auth token for this Webservice, scoped to the current user.

        :return: The auth token for this Webservice and when it should be refreshed after.
        :rtype: str, datetime
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        pass

    @abstractmethod
    def update(self, *args):
        """
        Update the Webservice parameters.

        This is an abstract method implemented by child classes of :class:`azureml.core.Webservice`.
        Possible parameters to update vary based on Webservice child type. For example, for Azure Container Instances
        webservices, see :func:`azureml.core.webservice.aci.AciWebservice.update` for specific parameters.

        :param args: Values to update.
        :type args: varies
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        pass

    def delete(self):
        """
        Delete this Webservice from its associated workspace.

        This function call is not asynchronous. The call runs until the resource is deleted.
        A :class:`azureml.exceptions.WebserviceException` is raised if there is a problem deleting the
        model from the Model Management Service.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        Webservice._check_for_webservice(self.workspace, self.name, self.compute_type,
                                         None, SERVICE_REQUEST_OPERATION_DELETE)

        headers = self._auth.get_authentication_header()
        params = {}

        resp = ClientBase._execute_func(get_requests_session().delete, self._mms_endpoint, headers=headers,
                                        params=params, timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code == 200:
            self.state = 'Deleting'
        elif resp.status_code == 202:
            self.state = 'Deleting'
            if 'Operation-Location' in resp.headers:
                operation_location = resp.headers['Operation-Location']
            else:
                raise WebserviceException('Missing response header key: Operation-Location', resp.status_code,
                                          logger=module_logger)
            create_operation_status_id = operation_location.split('/')[-1]
            operation_url = _get_mms_url(self.workspace) + '/operations/{}'.format(create_operation_status_id)

            self._operation_endpoint = operation_url
            self._wait_for_operation_to_complete(True, None)
        elif resp.status_code == 204:
            print('No service with name {} found to delete.'.format(self.name))
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    def serialize(self):
        """
        Convert this Webservice object into a JSON serialized dictionary.

        Use :func:`deserialize` to convert back into a Webservice object.

        :return: The JSON representation of this Webservice.
        :rtype: dict
        """
        created_time = self.created_time.isoformat() if self.created_time else None
        updated_time = self.updated_time.isoformat() if self.updated_time else None
        scoring_uri = getattr(self, 'scoring_uri', None)
        image_details = self.image.serialize() if self.image else None
        return {'name': self.name, 'description': self.description, 'tags': self.tags,
                'properties': self.properties, 'state': self.state, 'createdTime': created_time,
                'updatedTime': updated_time, 'error': self.error, 'computeType': self.compute_type,
                'workspaceName': self.workspace.name, 'imageId': self.image_id, 'imageDigest': self.image_digest,
                'imageDetails': image_details, 'scoringUri': scoring_uri, 'createdBy': self.created_by}

    @classmethod
    def deserialize(cls, workspace, webservice_payload):
        """
        Convert a Model Management Service response JSON object into a Webservice object.

        Will fail if the provided workspace is not the workspace the Webservice is registered under.

        :param cls: Indicates that this is a class method.
        :type cls:
        :param workspace: The workspace object the Webservice is registered under.
        :type workspace: azureml.core.Workspace
        :param webservice_payload: A JSON object to convert to a Webservice object.
        :type webservice_payload: dict
        :return: The Webservice representation of the provided JSON object.
        :rtype: azureml.core.Webservice
        """
        cls._validate_get_payload(webservice_payload)
        webservice = cls(None, None)
        webservice._initialize(workspace, webservice_payload)
        return webservice

    @classmethod
    def _validate_get_payload(cls, payload):
        """Validate the payload for this Webservice.

        :param payload:
        :type payload: dict
        :return:
        :rtype:
        """
        if 'computeType' not in payload:
            raise WebserviceException('Invalid payload for {} webservice, missing computeType:\n'
                                      '{}'.format(cls._webservice_type, payload), logger=module_logger)
        if payload['computeType'] != cls._webservice_type and cls._webservice_type != UNKNOWN_WEBSERVICE_TYPE:
            raise WebserviceException('Invalid payload for {} webservice, computeType is not {}":\n'
                                      '{}'.format(cls._webservice_type, cls._webservice_type, payload),
                                      logger=module_logger)
        for service_key in cls._expected_payload_keys:
            if service_key not in payload:
                raise WebserviceException('Invalid {} Webservice payload, missing "{}":\n'
                                          '{}'.format(cls._webservice_type, service_key, payload),
                                          logger=module_logger)


class ContainerResourceRequirements(object):
    """Defines the resource requirements for a container used by the Webservice.

    To specify autoscaling configuration, you will typically use the ``deploy_configuration``
    method of the :class:`azureml.core.webservice.aks.AksWebservice` class or the
    :class:`azureml.core.webservice.aci.AciWebservice` class.

    :var cpu: The number of CPU cores to allocate for this Webservice. Can be a decimal.
    :vartype cpu: float
    :var memory_in_gb: The amount of memory (in GB) to allocate for this Webservice. Can be a decimal.
    :vartype memory_in_gb: float
    :var cpu_limit: The max number of CPU cores this Webservice is allowed to use. Can be a decimal.
    :vartype cpu_limit: float
    :var memory_in_gb_limit: The max amount of memory (in GB) this Webservice is allowed to use. Can be a decimal.
    :vartype memory_in_gb_limit: float
    :var gpu: The number of GPU cores to allocate for this Webservice.
    :vartype gpu: int

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


class LivenessProbeRequirements(object):
    """Defines liveness probe time requirements for deployments of the Webservice.

    To specify autoscaling configuration, you will typically use the ``deploy_configuration`` or the ``update``
    method of the :class:`azureml.core.webservice.aks.AksWebservice` class.

    :var period_seconds: How often (in seconds) to perform the liveness probe. Defaults to 10 seconds.
        Minimum value is 1.
    :vartype period_seconds: int
    :var initial_delay_seconds: The number of seconds after the container has started before liveness probes are
        initiated.
    :vartype initial_delay_seconds: int
    :var timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 1 second.
        Minimum value is 1.
    :vartype timeout_seconds: int
    :var failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try ``failureThreshold``
        times before giving up. Defaults to 3. Minimum value is 1.
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
    :param timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 1 second.
        Minimum value is 1.
    :type timeout_seconds: int
    :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
        ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
    :type failure_threshold: int
    :param success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
        after having failed. Defaults to 1. Minimum value is 1.
    :type success_threshold: int
    """

    _expected_payload_keys = ['periodSeconds', 'initialDelaySeconds', 'timeoutSeconds',
                              'failureThreshold', 'successThreshold']

    def __init__(self, period_seconds, initial_delay_seconds, timeout_seconds, success_threshold, failure_threshold):
        """Initialize the liveness probe time requirements.

        :param period_seconds: How often (in seconds) to perform the liveness probe. Defaults to 10 seconds.
            Minimum value is 1.
        :type period_seconds: int
        :param initial_delay_seconds: The number of seconds after the container has started before liveness probes are
            initiated.
        :type initial_delay_seconds: int
        :param timeout_seconds: Number of seconds after which the liveness probe times out. Defaults to 1 second.
            Minimum value is 1.
        :type timeout_seconds: int
        :param failure_threshold: When a Pod starts and the liveness probe fails, Kubernetes will try
            ``failureThreshold`` times before giving up. Defaults to 3. Minimum value is 1.
        :type failure_threshold: int
        :param success_threshold: Minimum consecutive successes for the liveness probe to be considered successful
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
        :rtype: azureml.core.webservice.webservice.LivenessProbeRequirements
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


class AutoScaler(object):
    """Defines details for autoscaling configuration of a Kubernetes Webservice.

    To specify autoscaling configuration, you will typically use the ``deploy_configuration`` or the ``update``
    method of the :class:`azureml.core.webservice.aks.AksWebservice` class.

    :var autoscale_enabled: Indicates whether the AutoScaler is enabled or disabled.
    :vartype autoscale_enabled: bool
    :var max_replicas: The maximum number of containers for the Autoscaler to use.
    :vartype max_replicas: int
    :var min_replicas: The minimum number of containers for the Autoscaler to use.
    :vartype min_replicas: int
    :var refresh_period_seconds: How often the AutoScaler should attempt to scale the Webservice.
    :vartype refresh_period_seconds: int
    :var target_utilization: The target utilization (in percent out of 100) the AutoScaler should
        attempt to maintain for the Webservice.
    :vartype target_utilization: int

    :param autoscale_enabled: Indicates whether the AutoScaler is enabled or disabled.
    :type autoscale_enabled: bool
    :param max_replicas: The maximum number of containers for the Autoscaler to use.
    :type max_replicas: int
    :param min_replicas: The minimum number of containers for the Autoscaler to use.
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
        """Initialize the AutoScaler.

        :param autoscale_enabled: Indicates whether the AutoScaler is enabled or disabled.
        :type autoscale_enabled: bool
        :param max_replicas: The maximum number of containers for the Autoscaler to use.
        :type max_replicas: int
        :param min_replicas: The minimum number of containers for the Autoscaler to use.
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
        :rtype: azureml.core.webservice.webservice.AutoScaler
        """
        for payload_key in AutoScaler._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for autoScaler:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return AutoScaler(payload_obj['autoscaleEnabled'], payload_obj['maxReplicas'], payload_obj['minReplicas'],
                          payload_obj['refreshPeriodInSeconds'], payload_obj['targetUtilization'])


class DataCollection(object):
    """Defines data collection configuration for a Webservice.

    :var event_hub_enabled: Indicates whether event hub is enabled for the Webservice.
    :vartype event_hub_enabled: bool
    :var storage_enabled: Indicates whether data collection storage is enabled for the Webservice.
    :vartype storage_enabled: bool

    :param event_hub_enabled: Whether or not event hub is enabled for the Webservice
    :type event_hub_enabled: bool
    :param storage_enabled: Whether or not data collection storage is enabled for the Webservice
    :type storage_enabled: bool
    """

    _expected_payload_keys = ['eventHubEnabled', 'storageEnabled']

    def __init__(self, event_hub_enabled, storage_enabled):
        """Intialize the DataCollection object.

        :param event_hub_enabled: Whether or not event hub is enabled for the Webservice
        :type event_hub_enabled: bool
        :param storage_enabled: Whether or not data collection storage is enabled for the Webservice
        :type storage_enabled: bool
        """
        self.event_hub_enabled = event_hub_enabled
        self.storage_enabled = storage_enabled

    def serialize(self):
        """Convert this DataCollection object into a JSON serialized dictionary.

        :return: The JSON representation of this DataCollection.
        :rtype: dict
        """
        return {'eventHubEnabled': self.event_hub_enabled, 'storageEnabled': self.storage_enabled}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a DataCollection object.

        :param payload_obj: A JSON object to convert to a DataCollection object.
        :type payload_obj: dict
        :return: The DataCollection representation of the provided JSON object.
        :rtype: azureml.core.webservice.webservice.DataCollection
        """
        for payload_key in DataCollection._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for DataCollection:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return DataCollection(payload_obj['eventHubEnabled'], payload_obj['storageEnabled'])


class WebserviceDeploymentConfiguration(ABC):
    """Defines the base-class functionality for all Webservice deployment configuration objects.

    This class represents the configuration parameters for deploying a Webservice on a specific target.
    For example, to create deployment for Azure Kubernetes Service, use the ``deploy_configuration`` method
    of the :class:`azureml.core.webservice.aks.AksWebservice` class.

    :var azureml.core.webservice.Webservice.description: A description to give this Webservice.
    :vartype description: str
    :var azureml.core.webservice.Webservice.tags: A dictionary of key value tags to give this Webservice.
    :vartype tags: dict[str, str]
    :var azureml.core.webservice.Webservice.properties: A dictionary of key value properties to give this Webservice.
        These properties cannot be changed after deployment, however new key value pairs can be added.
    :vartype properties: dict[str, str]
    :var azureml.core.webservice.Webservice.primary_key: A primary auth key to use for this Webservice.
    :vartype primary_key: str
    :var azureml.core.webservice.Webservice.secondary_key: A secondary auth key to use for this Webservice.
    :vartype secondary_key: str
    :var azureml.core.webservice.Webservice.location: The Azure region to deploy this Webservice to.
    :vartype location: str

    :param type: The type of Webservice associated with this object.
    :type type: azureml.core.webservice.webservice.Webservice
    :param description: A description to give this Webservice.
    :type description: str
    :param tags: A dictionary of key value tags to give this Webservice.
    :type tags: dict[str, str]
    :param properties: A dictionary of key value properties to give this Webservice. These properties cannot
        be changed after deployment, however new key value pairs can be added.
    :type properties: dict[str, str]
    :param primary_key: A primary auth key to use for this Webservice.
    :type primary_key: str
    :param secondary_key: A secondary auth key to use for this Webservice.
    :type secondary_key: str
    :param location: The Azure region to deploy this Webservice to.
    :type location: str
    """

    def __init__(self, type, description=None, tags=None, properties=None, primary_key=None, secondary_key=None,
                 location=None):
        """Initialize the configuration object.

        :param type: The type of Webservice associated with this object.
        :type type: azureml.core.webservice.webservice.Webservice
        :param description: A description to give this Webservice.
        :type description: str
        :param tags: A dictionary of key value tags to give this Webservice.
        :type tags: dict[str, str]
        :param properties: A dictionary of key value properties to give this Webservice. These properties cannot
            be changed after deployment, however new key value pairs can be added.
        :type properties: dict[str, str]
        :param primary_key: A primary auth key to use for this Webservice.
        :type primary_key: str
        :param secondary_key: A secondary auth key to use for this Webservice.
        :type secondary_key: str
        :param location: The Azure region to deploy this Webservice to.
        :type location: str
        """
        self._webservice_type = type
        self.description = description
        self.tags = tags
        self.properties = properties
        self.primary_key = primary_key
        self.secondary_key = secondary_key
        self.location = location

    @abstractmethod
    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.WebserviceException` if validation fails.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        pass

    @abstractmethod
    def print_deploy_configuration(self):
        """Print the deployment configuration."""
        pass

    @classmethod
    def validate_image(cls, image):
        """Check that the image that is being deployed to the Webservice is valid.

        Raises a :class:`azureml.exceptions.WebserviceException` if validation fails.

        :param cls: Indicates that this is a class method.
        :type cls:
        :param image: The image that will be deployed to the webservice.
        :type image: azureml.core.Image
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if image is None:
            raise WebserviceException("Image is None", logger=module_logger)
        if image.creation_state != 'Succeeded':
            raise WebserviceException('Unable to create service with image {} in non "Succeeded" '
                                      'creation state.'.format(image.id), logger=module_logger)
        if image.image_flavor not in CLOUD_DEPLOYABLE_IMAGE_FLAVORS:
            raise WebserviceException('Deployment of {} images is not supported'.format(image.image_flavor),
                                      logger=module_logger)

    def _build_base_create_payload(self, name, environment_image_request):
        """Construct the base webservice creation payload.

        :param name:
        :type name: str
        :param environment_image_request:
        :type environment_image_request: dict
        :return:
        :rtype: dict
        """
        import copy
        from azureml._model_management._util import base_service_create_payload_template
        json_payload = copy.deepcopy(base_service_create_payload_template)
        json_payload['name'] = name
        json_payload['description'] = self.description
        json_payload['kvTags'] = self.tags

        properties = self.properties or {}
        properties.update(global_tracking_info_registry.gather_all())
        json_payload['properties'] = properties

        if self.primary_key:
            json_payload['keys']['primaryKey'] = self.primary_key
            json_payload['keys']['secondaryKey'] = self.secondary_key

        json_payload['computeType'] = self._webservice_type._webservice_type
        json_payload['environmentImageRequest'] = environment_image_request
        json_payload['location'] = self.location

        return json_payload


class WebServiceAccessToken(object):
    """
    Defines base functionality for retrieving the Access Token for deployed web services in Azure Machine Learning.

    :param access_token: The access token.
    :type token_type: str
    """

    _expected_payload_keys = ['accessToken', 'refreshAfter', 'expiryOn', 'tokenType']

    def __init__(self, access_token, token_type, refresh_after, expiry_on):
        """Create a new instance of WebServiceAccessToken.

        :param access_token: The access token.
        :type access_token: str
        :param refresh_after: Time after which access token should be fetched again.
        :type refresh_after: datetime
        :param expiry_on: Expiration time of the access token.
        :type expiry_on: datetime
        :param token_type: The type of access token.
        :type token_type: str
        """
        self.access_token = access_token
        self.refresh_after = datetime.fromtimestamp(refresh_after)
        self.expiry_on = datetime.fromtimestamp(expiry_on)
        self.token_type = token_type

    def serialize(self):
        """Convert this WeberviceAccessToken object into a JSON serialized dictionary.

        :return: The JSON representation of this WebServiceAccessToken object.
        :rtype: dict
        """
        return {
            'accessToken': self.access_token,
            'tokenType': self.token_type,
            'refreshAfter': self.refresh_after,
            'expiryOn': self.expiry_on
        }

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a WebServiceAccessToken object.

        :param payload_obj: A JSON object to convert to a WebServiceAccessToken object.
        :type payload_obj: dict
        :return: The WebServiceAccessToken representation of the provided JSON object.
        :rtype: azureml.core.webservice.webservice.WebServiceAccessToken
        """
        for payload_key in WebServiceAccessToken._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid webservice payload, missing {} for WebServiceAccessToken:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return WebServiceAccessToken(payload_obj['accessToken'], payload_obj['tokenType'], payload_obj['refreshAfter'],
                                     payload_obj['expiryOn'])
