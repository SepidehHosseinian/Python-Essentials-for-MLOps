# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for deploying machine learning models as local web service endpoints.

Deploying to a local web service is recommended for scenarios when you need to quickly deploy and
validate your model or you are testing a model that is under development. For more information,
see [Deploy a model to Notebook
VMs](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-local-container-notebook-vm).
"""

import datetime
import logging
import os
import pickle
import posixpath
import shutil
import tempfile
import json

from azureml.core.image.container import ContainerImageConfig
from azureml.core.model import Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice
from azureml.core.webservice.webservice import WebserviceDeploymentConfiguration
from azureml.exceptions import WebserviceException
from azureml._model_management._constants import DOCKER_IMAGE_APP_ROOT_DIR
from azureml._model_management._constants import DOCKER_IMAGE_HTTP_PORT
from azureml._model_management._constants import DOCKER_IMAGE_MQTT_PORT
from azureml._model_management._constants import LOCAL_WEBSERVICE_TYPE
from azureml._model_management._constants import MODEL_PACKAGE_ASSETS_DIR
from azureml._model_management._constants import MODEL_PACKAGE_MODELS_DIR
from azureml._model_management._constants import WEBSERVICE_SCORE_PATH
from azureml._model_management._constants import WEBSERVICE_SWAGGER_PATH
from azureml._model_management._util import add_sdk_to_requirements
from azureml._model_management._util import build_docker_image
from azureml._model_management._util import cleanup_container
from azureml._model_management._util import cleanup_docker_image
from azureml._model_management._util import connect_docker_network
from azureml._model_management._util import container_health_check
from azureml._model_management._util import container_scoring_call
from azureml._model_management._util import convert_parts_to_environment
from azureml._model_management._util import create_docker_container
from azureml._model_management._util import download_docker_build_context
from azureml._model_management._util import get_docker_client
from azureml._model_management._util import get_docker_container
from azureml._model_management._util import get_docker_containers
from azureml._model_management._util import get_docker_host_container
from azureml._model_management._util import get_docker_logs
from azureml._model_management._util import get_docker_network
from azureml._model_management._util import get_docker_port
from azureml._model_management._util import login_to_docker_registry
from azureml._model_management._util import make_mms_request
from azureml._model_management._util import start_docker_container
from azureml._model_management._util import write_dir_in_container
from azureml._model_management._util import write_file_in_container
from base64 import b64decode, b64encode
from functools import wraps

module_logger = logging.getLogger(__name__)


def _in_state(states):
    states = states if isinstance(states, (list, tuple)) else [states]

    def decorator(func):
        @wraps(func)
        def decorated(self, *args, **kwargs):
            if self.state not in states:
                raise WebserviceException('Cannot call {}() when service is {}.'.format(func.__name__, self.state),
                                          logger=module_logger)
            return func(self, *args, **kwargs)
        return decorated
    return decorator


class LocalWebservice(Webservice):
    """Represents a machine learning model deployed as a local web service endpoint.

    Deploying web services locally is useful for debugging and testing scenarios.

    .. remarks::

        The following code samples shows how to create a local Docker web service. See the notebook link for
        more details.

        .. code-block:: python

            from azureml.core.webservice import LocalWebservice

            # This is optional, if not provided Docker will choose a random unused port.
            deployment_config = LocalWebservice.deploy_configuration(port=6789)

            local_service = Model.deploy(ws, "test", [model], inference_config, deployment_config)

            local_service.wait_for_deployment()

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/deploy-to-local/register-model-deploy-local.ipynb


    :param workspace: The workspace object containing any Model objects that will be retrieved.
    :type workspace: azureml.core.Workspace
    :param name: The name of the Webservice object to retrieve.
    :type name: str
    :param must_exist: Whether the webservice must already exist when creating the in-memory object.
    :type must_exist: bool
    """

    _CONTAINER_LABEL_DEPLOYMENT_CONFIG = 'amlDeploymentConfig'
    _CONTAINER_LABEL_IMAGE_CONFIG = 'amlImageConfig'
    _CONTAINER_LABEL_INFERENCE_CONFIG = 'amlInferenceConfig'
    _CONTAINER_LABEL_IS_SERVICE = 'amlIsService'
    _CONTAINER_LABEL_MODEL_PATH_DIR = 'amlModelPathDir'
    _CONTAINER_LABEL_MODELS_LOCAL = 'amlModelsLocal'
    _CONTAINER_LABEL_MODELS_REMOTE = 'amlModelsRemote'
    _CONTAINER_LABEL_SERVICE_NAME = 'amlServiceName'
    _CONTAINER_LABEL_OVERRIDE_CONFIG = 'amlOverrideConfig'

    _CONTAINER_ENVVAR_MODEL_PATH = "AZUREML_MODEL_DIR"

    NETWORK_NAME = 'azureml-local'

    STATE_DELETED = 'deleted'
    STATE_DEPLOYING = 'deploying'
    STATE_FAILED = 'failed'
    STATE_RUNNING = 'running'
    STATE_UNKNOWN = 'unknown'

    _webservice_type = LOCAL_WEBSERVICE_TYPE

    # Ignore Webservice's magic __new__(), which assumes there is a cloud
    # representation of the service.
    def __new__(cls, *args, **kwargs):
        """
        Local webservice constructor.

        LocalWebservice constructor is used to retrieve a local representation of a LocalWebservice object associated
        with the provided workspace.

        :param workspace: The workspace object containing any Model objects that will be retrieved.
        :type workspace: azureml.core.Workspace
        :param name: The name of the LocalWebservice object to retrieve.
        :type name: str
        :param must_exist: Whether the webservice must already exist when creating the in-memory object.
        :type must_exist: bool
        :return: An instance of LocalWebservice.
        :rtype: azureml.core.webservice.LocalWebservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        return super(Webservice, cls).__new__(cls)

    def __init__(self, workspace, name, must_exist=True):
        """
        Local webservice constructor.

        LocalWebservice constructor is used to retrieve a local representation of a LocalWebservice object associated
        with the provided workspace.

        :param workspace: The workspace object containing any Model objects that will be retrieved.
        :type workspace: azureml.core.Workspace
        :param name: The name of the LocalWebservice object to retrieve.
        :type name: str
        :param must_exist: Whether the webservice must already exist when creating the in-memory object.
        :type must_exist: bool
        :return: An instance of LocalWebservice.
        :rtype: azureml.core.webservice.LocalWebservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        self.name = name
        self.compute_type = LOCAL_WEBSERVICE_TYPE
        self.workspace = workspace
        self.error = None
        self.state = None
        self.image_id = None
        self.tags = None
        self.properties = None
        self.created_time = datetime.datetime.now()
        self.updated_time = datetime.datetime.now()

        self._docker_client = get_docker_client()

        self._host_container = get_docker_host_container(self._docker_client)

        self._image_config = None
        self._inference_config = None

        if self._host_container is not None:
            self._host_network = get_docker_network(self._docker_client, LocalWebservice.NETWORK_NAME)

        self.update_deployment_state(must_exist=must_exist)

    def __repr__(self):
        """Return the string representation of the LocalWebservice object.

        :return: String representation of the LocalWebservice object
        :rtype: str
        """
        return super().__repr__()

    @_in_state([STATE_DEPLOYING, STATE_FAILED, STATE_RUNNING])
    def delete(self, delete_cache=True, delete_image=False, delete_volume=True):
        """
        Delete this LocalWebservice from the local machine.

        This function call is not asynchronous; it runs until the service is deleted.

        :param delete_cache: Whether to delete temporary files cached for the service. (Default: True)
        :type delete_cache: bool
        :param delete_image: Whether to delete the service's Docker image. (Default: False)
        :type delete_image: bool
        :param delete_volume: Whether to delete the service's Docker volume. (Default: True)
        :type delete_volume: bool
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        cleanup_container(self._container)

        if delete_cache:
            self._context_dir = None
            self._models_dir = None

        if delete_image:
            cleanup_docker_image(self._docker_client, self._container.image.id)

        self.state = LocalWebservice.STATE_DELETED

    @staticmethod
    def deploy_configuration(port=None):
        """
        Create a configuration object for deploying a local Webservice.

        :param port: The local port on which to expose the service's HTTP endpoint.
        :type port: int
        :return: A configuration object to use when deploying a Webservice object.
        :rtype: azureml.core.webservice.local.LocalWebserviceDeploymentConfiguration
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        return LocalWebserviceDeploymentConfiguration(port)

    def deploy_to_cloud(self, name=None, deployment_config=None, deployment_target=None):
        """
        Deploy a Webservice based on the LocalWebservice's configuration.

        :param name: The name to give the deployed service. Must be unique to the workspace.
        :type name: str
        :param deployment_config: A WebserviceDeploymentConfiguration used to configure the webservice. If one is not
            provided, an empty configuration object will be used based on the desired target.
        :type deployment_config: azureml.core.webservice.webservice.WebserviceDeploymentConfiguration
        :param deployment_target: A :class:`azureml.core.ComputeTarget` to which to deploy the Webservice. As
            ACI has no associated :class:`azureml.core.ComputeTarget`, leave this parameter as None to deploy to ACI.
        :type deployment_target: azureml.core.ComputeTarget
        :return: A Webservice object corresponding to the deployed webservice.
        :rtype: azureml.core.Webservice
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if not self._deployable:
            raise WebserviceException('The LocalWebservice is not deployable (state = {}). Recreate it, or use '
                                      'the Model.deploy() method directly.'.format(self.state), logger=module_logger)

        models = [model['local_path'] for model in self._models_local] + self._models_remote

        if self._image_config is not None:  # pragma: no cover
            return Webservice.deploy_from_model(workspace=self.workspace,
                                                name=name or self.name,
                                                models=models,
                                                image_config=self._image_config,
                                                deployment_config=deployment_config,
                                                deployment_target=deployment_target)
        else:
            return Model.deploy(workspace=self.workspace,
                                name=name or self.name,
                                models=models,
                                inference_config=self._inference_config,
                                deployment_config=deployment_config,
                                deployment_target=deployment_target)

    @classmethod
    def deserialize(cls, workspace, webservice_payload):
        """
        Convert a Model Management Service response JSON object into a Webservice object.

        .. note::
            Not supported for LocalWebservice.

        :param cls:
        :type cls:
        :param workspace: The workspace object the Webservice is registered under.
        :type workspace: azureml.core.Workspace
        :param webservice_payload: A JSON object to convert to a Webservice object.
        :type webservice_payload: dict
        :raises: azureml.exceptions.NotImplementedError
        """
        raise NotImplementedError('Local web services cannot be deserialized.')

    def get_keys(self):
        """
        Retrieve auth keys for this Webservice.

        .. note::
            Not supported for LocalWebservice.

        :return: The auth keys for this Webservice.
        :raises: azureml.exceptions.NotImplementedError
        """
        raise NotImplementedError('Local web services do not support auth keys.')

    @_in_state([STATE_DEPLOYING, STATE_FAILED, STATE_RUNNING])
    def get_logs(self, num_lines=5000, raw=False):
        """
        Retrieve logs for this LocalWebservice.

        :param num_lines: The maximum number of log lines to retrieve. (Default: 5000)
        :type num_lines: int
        :param raw: Return the raw Docker container output without attempting to format it. (Default: False)
        :type raw: bool
        :return: The logs for this LocalWebservice.
        :rtype: str
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        return get_docker_logs(self._container, num_lines=num_lines)

    def get_token(self):
        """
        Retrieve auth token for this Webservice, scoped to the current user.

        .. note::
            Not supported for LocalWebservice.

        :return: The auth token for this Webservice and when it should be refreshed after.
        :rtype: str, datetime
        :raises: azureml.exceptions.NotImplementedError
        """
        raise NotImplementedError("Local web services do not support Token Authentication.")

    @staticmethod
    def list(workspace, model_name=None, model_id=None, all=None):
        """
        List the LocalWebservices associated with the corresponding Workspace.

        The results returned can be filtered using parameters.

        :param workspace: The Workspace object associated with the LocalWebservices.
        :type workspace: azureml.core.Workspace
        :param model_name: Filter list to only include LocalWebservices deployed with the specific model name.
        :type model_name: str
        :param model_id: Filter list to only include LocalWebservices deployed with the specific model ID.
        :type model_id: str
        :param all: Show all services. Only running services are shown by default.
        :type all: bool
        :return: A filtered list of LocalWebservices associated with the provided Workspace.
        :rtype: builtin.list[azureml.core.webservice.LocalWebservice]
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        services = []

        for container in get_docker_containers(get_docker_client(),
                                               label_name=LocalWebservice._CONTAINER_LABEL_IS_SERVICE,
                                               label_value=str(True),
                                               all=all):
            try:
                name = container.labels[LocalWebservice._CONTAINER_LABEL_SERVICE_NAME]
            except KeyError:
                continue

            service = LocalWebservice(workspace, name)

            try:
                if (model_name is None
                        or any(model['name'] == model_name for model in service._models_local)
                        or any(model.name == model_name for model in service._models_remote)) \
                        and (model_id is None or any(model.id == model_id for model in service._models_remote)):
                    services.append(service)
            except AttributeError:
                continue

        return services

    @property
    def port(self):
        """
        Get the local webservice port.

        :return: Port number.
        :rtype: int
        """
        return getattr(self, '_port', None)

    def regen_key(self, key):
        """
        Regenerate one of the Webservice's keys.

        .. note::
            Not supported for LocalWebservice.

        :param key: Which key to regenerate. Options are 'Primary' or 'Secondary'
        :type key: str
        :raises: NotImplementedError
        """
        raise NotImplementedError('Local web services do not support auth keys.')

    @_in_state([STATE_DEPLOYING, STATE_FAILED, STATE_RUNNING])
    def reload(self, wait=False):
        """
        Reload the LocalWebservice's execution script and dependencies.

        This restarts the service's container with copies of updated assets, including the execution script and local
        dependencies, but it does not rebuild the underlying image. Accordingly, changes to Conda/pip dependencies or
        custom Docker steps will not be reflected in the reloaded LocalWebservice. To handle those changes call
        the :meth:`azureml.core.webservice.local.LocalWebservice.update` method instead.

        :param wait: Wait for the service's container to reach a healthy state. (Default: False)
        :type wait: bool
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        self._run_container(wait=wait)

    @_in_state([STATE_DEPLOYING, STATE_RUNNING])
    def run(self, input_data):
        """
        Call this LocalWebservice with the provided input.

        :param input_data: The input with which to call the LocalWebservice.
        :type input_data: varies
        :return: The result of calling the LocalWebservice.
        :rtype: varies
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if self.state == LocalWebservice.STATE_DEPLOYING:
            self.wait_for_deployment()

        return container_scoring_call(self._port, input_data, self._container, self._internal_base_url,
                                      cleanup_on_failure=False, score_url=self.scoring_uri)

    @property
    def scoring_uri(self):
        """
        Get the local webservice scoring URI.

        :return: Scoring URI.
        :rtype: str
        """
        if hasattr(self, '_base_url'):
            scorePath = WEBSERVICE_SCORE_PATH
            if getattr(self, '_container_override_config', None) is not None:
                overrideScorePath = self._container_override_config["scorePath"]
                if overrideScorePath is not None:
                    scorePath = overrideScorePath
            return "{}/{}".format(self._base_url.rstrip("/"), scorePath.lstrip("/"))
        else:
            return None

    def serialize(self):
        """
        Convert this Webservice object into a JSON-serialized dictionary.

        :return: Serialized representation of the Webservice object.
        :rtype: dict
        """
        created_time = self.created_time.isoformat() if self.created_time else None
        updated_time = self.updated_time.isoformat() if self.updated_time else None
        return {'name': self.name, 'tags': self.tags, 'properties': self.properties, 'state': self.state,
                'createdTime': created_time, 'updatedTime': updated_time, 'error': self.error,
                'computeType': self.compute_type, 'workspaceName': self.workspace.name, 'imageId': self.image_id,
                'scoringUri': self.scoring_uri, 'swaggerUri': self.swagger_uri, 'port': self.port}

    @property
    def swagger_uri(self):
        """
        Get the local webservice Swagger URI.

        :return: Swagger URI.
        :rtype: str
        """
        if hasattr(self, '_base_url'):
            return self._base_url + WEBSERVICE_SWAGGER_PATH
        else:
            return None

    @_in_state([STATE_DEPLOYING, STATE_FAILED, STATE_RUNNING, STATE_UNKNOWN])
    def update(self, models=None, image_config=None, deployment_config=None, wait=False, inference_config=None):
        """
        Update the LocalWebservice with provided properties.

        Values left as None will remain unchanged in this LocalWebservice.

        :param models: A new list of models contained in the LocalWebservice.
        :type models: builtin.list[azureml.core.Model]
        :param image_config: Image configuration options to apply to the LocalWebservice.
        :type image_config: azureml.core.image.container.ContainerImageConfig
        :param deployment_config: Deployment configuration options to apply to the LocalWebservice.
        :type deployment_config: azureml.core.webservice.local.LocalWebserviceDeploymentConfiguration
        :param inference_config: An InferenceConfig object used to provide the required model deployment properties.
        :type inference_config: azureml.core.model.InferenceConfig
        :param wait: Wait for the service's container to reach a healthy state. (Default: False)
        :type wait: bool
        :return:
        :rtype: None
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if image_config and not isinstance(image_config, ContainerImageConfig):  # pragma: no cover
            raise WebserviceException('Error, provided image configuration must be of type ContainerImageConfig '
                                      'in order to update a local service.', logger=module_logger)

        if deployment_config and not isinstance(deployment_config, LocalWebserviceDeploymentConfiguration):
            raise WebserviceException('Error, provided deployment configuration must be of type '
                                      'LocalWebserviceDeploymentConfiguration in order to update a local service.',
                                      logger=module_logger)

        if inference_config is not None and not isinstance(inference_config, InferenceConfig):
            raise WebserviceException('Error, provided inference configuration must be of type InferenceConfig '
                                      'in order to update a local service.', logger=module_logger)

        if image_config is not None and inference_config is not None:  # pragma: no cover
            raise WebserviceException('Error, an image configuration and an inference configuration cannot be '
                                      'provided at the same time.', logger=module_logger)

        # Update our cached configuration
        if models is not None:
            self._models_dir = None
            self._models_local, self._models_remote = LocalWebservice._identify_models(self.workspace, models)

        if image_config is not None:  # pragma: no cover
            self._image_config = image_config
            self._inference_config = None

        if deployment_config is not None:
            self._deployment_config = deployment_config

        if inference_config is not None:
            inference_config, _ = convert_parts_to_environment(self.name, inference_config)

            if inference_config.environment:
                # EMS and Model.package()-based path
                self._image_config = None
                self._inference_config = inference_config
            else:  # pragma: no cover
                # Old runtime and ImageConfig-based path
                self._image_config = inference_config._convert_to_image_conf_for_local()
                self._inference_config = None

        self._deployable |= (self.workspace is not None
                             and self._models_local is not None
                             and self._models_remote is not None)

        # Apply the changes
        self._collect_models()
        self._generate_docker_context()
        self._build_image()
        self._run_container(wait=wait)

    def update_deployment_state(self, must_exist=False):
        """
        Refresh the current state of the in-memory object.

        Perform an in-place update of the properties of the object based on the current state of the corresponding
        local Docker container.

        :param must_exist: Whether the webservice must already exist when creating the in-memory object.
        :type must_exist: bool
        """
        # Look for an existing instance
        container = LocalWebservice._get_container(self._docker_client,
                                                   self.workspace,
                                                   self.name,
                                                   must_exist=must_exist)

        self.updated_time = datetime.datetime.now()

        if container:
            self._container = container
            self._image = container.image

            self.state = LocalWebservice.STATE_RUNNING

            try:
                self._base_url, self._internal_base_url, self._port = self._generate_base_urls()
            except Exception:
                self.state = LocalWebservice.STATE_FAILED

            try:
                self._deployment_config = LocalWebservice._get_container_deployment_config(container)

                self._image_config = LocalWebservice._get_container_image_config(container)
                self._inference_config = LocalWebservice._get_container_inference_config(container)

                self._model_path_dir = LocalWebservice._get_container_model_path_dir(container)
                self._models_dir = None
                self._models_local = LocalWebservice._get_container_models_local(container)
                self._models_remote = LocalWebservice._get_container_models_remote(self.workspace, container)
                self._container_override_config = LocalWebservice._get_container_override_config(container)

                self._deployable = True
            except Exception:
                self._deployable = False
                self.state = LocalWebservice.STATE_FAILED
        else:
            self.state = LocalWebservice.STATE_UNKNOWN

            self._deployable = False

    @_in_state([STATE_DEPLOYING, STATE_RUNNING])
    def wait_for_deployment(self, show_output=False):
        """
        Poll the running LocalWebservice deployment.

        :param show_output: Option to print more verbose output. (Default: False)
        :type show_output: bool
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        try:
            container_health_check(self._port,
                                   self._container,
                                   health_url=self._get_health_url(),
                                   cleanup_if_failed=False)

            self.state = LocalWebservice.STATE_RUNNING

            print('Local webservice is {} at {}'.format(self.state, self._base_url))
        except Exception:
            self.state = LocalWebservice.STATE_FAILED
            raise

    def _build_image(self):
        self._login_to_registry()
        self._image = build_docker_image(self._docker_client,
                                         self._context_dir.name,
                                         self.name,
                                         pull=True)

    def _collect_models(self):
        # Re-use already-collected models
        if self._models_dir is not None:
            return

        self._model_path_dir = ''
        self._models_dir = tempfile.TemporaryDirectory(prefix='azureml_')

        model_dir_posix = ''

        # Local models
        for model in self._models_local:
            model_dir_posix = posixpath.join(model['name'], '0')
            model_dir = os.path.join(self._models_dir.name, model['name'], '0')
            model_dest = os.path.join(model_dir, model['image_path'])

            print('Copying local model {} to {}'.format(model['local_path'], model_dir))

            if os.path.isdir(model['local_path']):
                shutil.copytree(model['local_path'], model_dest)
            else:
                os.makedirs(model_dir, exist_ok=True)
                shutil.copyfile(model['local_path'], model_dest)

        # Registered models
        for model in self._models_remote:
            model_dir_posix = posixpath.join(model.name, str(model.version))
            model_dir = os.path.join(self._models_dir.name, model.name, str(model.version))

            print('Downloading model {} to {}'.format(model.id, model_dir))

            model.download(target_dir=model_dir)

        # If single-model deployment, set the model path
        if len(self._models_local) + len(self._models_remote) == 1:
            self._model_path_dir = model_dir_posix

    @staticmethod
    def _construct_base_url(port):
        return 'http://localhost:{}'.format(port)

    @staticmethod
    def _construct_container_name(service_name):
        return service_name

    @staticmethod
    def _construct_internal_base_url(alias):
        return 'http://{}:{}'.format(alias, DOCKER_IMAGE_HTTP_PORT)

    @staticmethod
    def _copy_files_with_overwrite(src, dst):
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        elif os.path.isdir(src) and os.path.isdir(dst):
            for entry in os.listdir(src):
                LocalWebservice._copy_files_with_overwrite(os.path.join(src, entry), os.path.join(dst, entry))
        elif os.path.isfile(src) and os.path.isfile(dst):
            shutil.copy2(src, dst)
        else:
            raise FileExistsError('Cannot copy {} to {}: one is a folder and the other is a file.'.format(src, dst))

    @staticmethod
    def _copy_local_asset(container, local_path, container_rel_path=None):
        container_path_prefix = LocalWebservice._get_image_aml_app_root(container.image)
        container_rel_path = container_rel_path or os.path.basename(local_path.rstrip(os.sep))
        container_path = posixpath.join(container_path_prefix, container_rel_path)

        if os.path.isdir(local_path):
            write_dir_in_container(container, container_path, local_path)
        else:
            with open(local_path, 'rb') as f:
                write_file_in_container(container, container_path, f.read())

    @staticmethod
    def _deploy(workspace, name, models, image_config=None, deployment_config=None, wait=False, inference_config=None):
        """
        Deploy the Webservice.

        :param name:
        :type name: str
        :param models:
        :type models: builtin.list[azureml.core.Model]
        :param image_config:
        :type image_config: azureml.core.image.container.ContainerImageConfig
        :param deployment_config:
        :type deployment_config: azureml.core.webservice.LocalWebserviceDeploymentConfiguration
        :param wait:
        :type wait: bool
        :param inference_config:
        :type inference_config: azureml.core.model.InferenceConfig
        :return:
        :rtype: LocalDebuggingWebservice
        """
        if image_config and not isinstance(image_config, ContainerImageConfig):  # pragma: no cover
            raise WebserviceException('Error, provided image configuration must be of type ContainerImageConfig '
                                      'in order to deploy a local service.', logger=module_logger)

        if not deployment_config:
            deployment_config = LocalWebservice.deploy_configuration()
        elif not isinstance(deployment_config, LocalWebserviceDeploymentConfiguration):
            raise WebserviceException('Error, provided deployment configuration must be of type '
                                      'LocalWebserviceDeploymentConfiguration in order to deploy a local service.',
                                      logger=module_logger)

        if inference_config and not isinstance(inference_config, InferenceConfig):
            raise WebserviceException('Error, provided inference configuration must be of type InferenceConfig '
                                      'in order to deploy a local service.', logger=module_logger)

        service = LocalWebservice(workspace, name, must_exist=False)
        service.update(models=models,
                       image_config=image_config,
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       wait=wait)
        return service

    def _get_internal_port(self):
        if getattr(self, '_container_override_config', None) is not None:
            port = self._container_override_config.get("internalTargetPort")
            if port is not None:
                return port
        return DOCKER_IMAGE_HTTP_PORT

    def _get_health_url(self):
        if getattr(self, '_container_override_config', None) is not None:
            health_url = self._container_override_config.get("healthPath")
            if health_url is not None:
                return "{}/{}".format(self._internal_base_url, health_url.lstrip("/"))
        return self._internal_base_url

    def _generate_base_urls(self):
        if self._host_container is not None:
            connect_docker_network(self._host_network, self._host_container)
            connect_docker_network(self._host_network, self._container, aliases=[self._container.short_id])

        port = get_docker_port(self._docker_client, self.name, self._container, self._get_internal_port())
        base_url = LocalWebservice._construct_base_url(port)

        if self._host_container is not None:
            internal_base_url = LocalWebservice._construct_internal_base_url(self._container.short_id)
        else:
            internal_base_url = base_url

        return base_url, internal_base_url, port

    def _generate_docker_context(self):
        """
        Generate Dockerfile according to the LocalWebservice's configuration.

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        print('Generating Docker build context.')

        self._context_dir = tempfile.TemporaryDirectory(prefix='azureml_')

        assets_dir = os.path.join(self._context_dir.name, MODEL_PACKAGE_ASSETS_DIR)

        if self._image_config is not None:  # pragma: no cover
            # MMS-generated files
            model_ids = [model.id for model in self._models_remote]
            json_payload = self._image_config.create_local_debug_payload(self.workspace, model_ids)

            response = make_mms_request(self.workspace, 'POST', '/images/generateDockerBuildContext', json_payload)
            manifest = response.json()

            download_docker_build_context(self.workspace, manifest, self._context_dir.name)

            # Local user assets
            os.makedirs(assets_dir, exist_ok=True)

            def add_local_asset(local_path, image_path=None):
                image_path = image_path or os.path.basename(local_path.rstrip(os.sep))
                asset_path = os.path.join(assets_dir, image_path)

                if os.path.isdir(local_path):
                    shutil.copytree(local_path, asset_path)
                else:
                    os.makedirs(os.path.dirname(asset_path), exist_ok=True)
                    shutil.copyfile(local_path, asset_path)

            if self._image_config.conda_file:
                add_local_asset(self._image_config.conda_file)

            if self._image_config.schema_file:
                add_local_asset(self._image_config.schema_file)

            if self._image_config.dependencies:
                for dependency in self._image_config.dependencies:
                    add_local_asset(dependency)

            add_local_asset(add_sdk_to_requirements(), 'requirements.txt')

            self._registry = self._image_config.base_image_registry
        else:
            # MMS-generated files
            models = self._models_remote + self._models_local if self._inference_config is None else []

            package = Model.package(self.workspace, models, self._inference_config, generate_dockerfile=True)
            package.wait_for_creation(show_output=True)
            package.save(self._context_dir.name)

            self._container_override_config = LocalWebservice._load_container_override_config(self._context_dir.name)
            self._registry = package.get_container_registry()

        # Models from the models cache
        if self._models_dir is not None:
            LocalWebservice._copy_files_with_overwrite(self._models_dir.name,
                                                       os.path.join(assets_dir, MODEL_PACKAGE_MODELS_DIR))

    @staticmethod
    def _get_container(docker_client, workspace, name, must_exist=True):
        if not workspace or not name:
            return None

        try:
            container_name = LocalWebservice._construct_container_name(name)
            return get_docker_container(docker_client, container_name, all=True, limit=1)
        except WebserviceException as e:
            if not must_exist and 'WebserviceNotFound' in e.message:
                return None
            raise WebserviceException(e.message, logger=module_logger)

    @staticmethod
    def _get_container_deployment_config(container):
        deployment_config = pickle.loads(b64decode(
            container.labels[LocalWebservice._CONTAINER_LABEL_DEPLOYMENT_CONFIG]))
        deployment_config.validate_configuration()
        return deployment_config

    @staticmethod
    def _get_container_image_config(container):
        image_config = pickle.loads(b64decode(container.labels[LocalWebservice._CONTAINER_LABEL_IMAGE_CONFIG]))
        if image_config is not None:
            image_config.validate_configuration()
        return image_config

    @staticmethod
    def _get_container_inference_config(container):
        inference_config = pickle.loads(b64decode(container.labels[LocalWebservice._CONTAINER_LABEL_INFERENCE_CONFIG]))
        if inference_config is not None:
            inference_config.validate_configuration()
        return inference_config

    @staticmethod
    def _get_container_model_path_dir(container):
        return container.labels[LocalWebservice._CONTAINER_LABEL_MODEL_PATH_DIR]

    @staticmethod
    def _get_container_models_local(container):
        return pickle.loads(b64decode(container.labels[LocalWebservice._CONTAINER_LABEL_MODELS_LOCAL]))

    @staticmethod
    def _get_container_models_remote(workspace, container):
        model_ids = pickle.loads(b64decode(container.labels[LocalWebservice._CONTAINER_LABEL_MODELS_REMOTE]))
        return [Model(workspace, id=model_id) for model_id in model_ids]

    @staticmethod
    def _get_image_aml_app_root(image):
        try:
            envs = image.attrs['Config']['Env']
            return next(e for e in envs if e.startswith('AML_APP_ROOT=')).split('=', 1)[1]
        except Exception:
            return DOCKER_IMAGE_APP_ROOT_DIR

    @staticmethod
    def _get_container_override_config(container):
        return pickle.loads(b64decode(container.labels[LocalWebservice._CONTAINER_LABEL_OVERRIDE_CONFIG]))

    @staticmethod
    def _load_container_override_config(dir):
        override_config_file = os.path.join(dir, "container_override_config.json")
        if os.path.exists(override_config_file):
            with open(override_config_file) as f:
                return json.load(f)
        return None

    @staticmethod
    def _identify_models(workspace, models):
        local = []
        remote = []

        for model in models:
            if isinstance(model, str):
                try:
                    # Registered model; behave like a normal image build.
                    registered_model = Model(workspace, id=model)
                    remote.append(registered_model)
                except WebserviceException as e:
                    # Unregistered model; keep it local for the local debugging.
                    if 'ModelNotFound' in e.message:
                        model_path = os.path.basename(model.rstrip(os.sep))
                        model_name = model_path[:30]

                        local.append({'name': model_name, 'image_path': model_path, 'local_path': model})
                    else:
                        raise WebserviceException(e.message, logger=module_logger)
            elif isinstance(model, Model):
                remote.append(model)
            else:
                raise NotImplementedError('Models must either be of type azureml.core.model.Model or a str '
                                          'path to a file or folder.')

        return local, remote

    def _login_to_registry(self):
        # Base image credentials are optional
        if not self._registry:
            return

        if not self._registry.address and not self._registry.username and not self._registry.password:
            return

        if not self._registry.address or not self._registry.username or not self._registry.password:
            module_logger.warning('Base image registry should include all of address, username, and password.')

        login_to_docker_registry(self._docker_client, self._registry.username, self._registry.password,
                                 self._registry.address)

    def _run_container(self, wait):
        # Clean up the old container/image.
        if getattr(self, '_container', None) is not None:
            old_image_is_dangling = len(self._container.image.tags) == 0
            self.delete(delete_cache=False, delete_image=old_image_is_dangling)

        # Start up the new container.
        container_name = LocalWebservice._construct_container_name(self.name)

        container_override_config = getattr(self, '_container_override_config', None)
        labels = {
            LocalWebservice._CONTAINER_LABEL_DEPLOYMENT_CONFIG:
                b64encode(pickle.dumps(self._deployment_config)).decode('utf-8'),
            LocalWebservice._CONTAINER_LABEL_IMAGE_CONFIG:
                b64encode(pickle.dumps(self._image_config)).decode('utf-8'),
            LocalWebservice._CONTAINER_LABEL_INFERENCE_CONFIG:
                b64encode(pickle.dumps(self._inference_config)).decode('utf-8'),
            LocalWebservice._CONTAINER_LABEL_IS_SERVICE: str(True),
            LocalWebservice._CONTAINER_LABEL_MODEL_PATH_DIR: self._model_path_dir,
            LocalWebservice._CONTAINER_LABEL_MODELS_LOCAL:
                b64encode(pickle.dumps(self._models_local)).decode('utf-8'),
            LocalWebservice._CONTAINER_LABEL_MODELS_REMOTE:
                b64encode(pickle.dumps([model.id for model in self._models_remote])).decode('utf-8'),
            LocalWebservice._CONTAINER_LABEL_SERVICE_NAME: self.name,
            LocalWebservice._CONTAINER_LABEL_OVERRIDE_CONFIG: b64encode(pickle.dumps(container_override_config))
                .decode('utf-8'),
        }

        port = self._get_internal_port()
        ports = {
            port: ('127.0.0.1', self._deployment_config.port),  # None assigns a random port
            DOCKER_IMAGE_MQTT_PORT: ('127.0.0.1',),
        }

        environment = {
            LocalWebservice._CONTAINER_ENVVAR_MODEL_PATH:
                posixpath.join(MODEL_PACKAGE_MODELS_DIR, self._model_path_dir),
        }

        if self._inference_config \
                and self._inference_config.environment \
                and self._inference_config.environment.environment_variables:
            for k, v in self._inference_config.environment.environment_variables.items():
                environment[k] = v

        container = create_docker_container(self._docker_client,
                                            self.name,
                                            container_name,
                                            environment=environment,
                                            labels=labels,
                                            ports=ports)

        # Copy local assets into the AML working directory.
        if self._image_config is not None:  # pragma: no cover
            LocalWebservice._copy_local_asset(container, self._image_config.execution_script,
                                              container_rel_path='main.py')
            LocalWebservice._copy_local_asset(container, self._image_config.execution_script)

            if self._image_config.schema_file is not None:
                LocalWebservice._copy_local_asset(container, self._image_config.schema_file)

            if self._image_config.dependencies is not None:
                for dependency in self._image_config.dependencies:
                    LocalWebservice._copy_local_asset(container, dependency)
        elif self._inference_config is not None:
            if self._inference_config.source_directory:
                LocalWebservice._copy_local_asset(container, os.path.realpath(self._inference_config.source_directory))
            else:
                LocalWebservice._copy_local_asset(container, self._inference_config.entry_script,
                                                  container_rel_path='main.py')
                LocalWebservice._copy_local_asset(container, self._inference_config.entry_script)

        # Engage!
        start_docker_container(container)

        # Record the state of the deployment.
        self._container = container
        self.state = LocalWebservice.STATE_DEPLOYING

        # Make sure we can talk with the container.
        self._base_url, self._internal_base_url, self._port = self._generate_base_urls()

        if wait:
            self.wait_for_deployment()


class LocalWebserviceDeploymentConfiguration(WebserviceDeploymentConfiguration):
    """Defines a deployment configuration object for a web service endpoint deployed locally.

    Create a LocalWebserviceDeploymentConfiguration object using the
    :meth:`azureml.core.webservice.local.LocalWebservice.deploy_configuration` method of the
    :class:`azureml.core.webservice.local.LocalWebservice` class.

    :param port: The local port on which to expose the service's HTTP endpoint. Acceptable values [1, 65535].
    :type port: int
    """

    _webservice_type = LocalWebservice

    def __init__(self, port=None):
        """
        Create a configuration object for deploying a local service for debugging.

        :param port: The local port on which to expose the service's HTTP endpoint.
        :type port: int
        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        super(LocalWebserviceDeploymentConfiguration, self).__init__(LocalWebservice)

        self.port = port

        self.validate_configuration()

    def validate_configuration(self):
        """
        Check that the specified configuration values are valid.

        Will raise an :class:`azureml.exceptions.WebserviceException` if the specified port in not in the range
        [1, 65535].

        :raises: :class:`azureml.exceptions.WebserviceException`
        """
        if self.port and not (isinstance(self.port, int) and 1 <= self.port <= 65535):
            raise WebserviceException('Invalid configuration, port must be an integer in range [1, 65535]',
                                      logger=module_logger)

    def print_deploy_configuration(self):
        """Print the deployment configuration."""
        pass
