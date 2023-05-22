# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""**ServiceContext** container for scope and authentication of a workspace."""
import copy
import logging
import os
import requests
import six

from azureml._async.worker_pool import WorkerPool
from azureml._base_sdk_common.service_discovery import get_service_url
from azureml._logging.chained_identity import ChainedIdentity
from .rest_client import RestClient

DEVLEOPER_URL_TEMPLATE = "AZUREML_DEV_URL_{}"

module_logger = logging.getLogger(__name__)


class _ServiceKeys(object):
    """Enum of keys for retrieving service urls from discovery service."""
    API_DISCOVERY_KEY = "api"
    EXPERIMENTATION_DISCOVERY_KEY = "experimentation"
    HISTORY_DISCOVERY_KEY = "history"
    HYPERDRIVE_DISCOVERY_KEY = "hyperdrive"
    MMS_DISCOVERY_KEY = "modelmanagement"
    STUDIO_DISCOVERY_KEY = "studio"

    # For datastore:
    # DATASTORE = "datastore"

    API = "api"
    ARTIFACTS = "artifacts"
    ASSETS = "assets"
    CREDENTIAL = "credential"
    ENVIRONMENT = "environment"
    EXECUTION = "execution"
    EXPERIMENTATION = "experimentation"
    HYPERDRIVE = "hyperdrive"
    JASMINE = "jasmine"
    METRICS = "metrics"
    MLFLOW = "mlflow"
    MODELMANAGEMENT = "modelmanagement"
    PIPELINES = "pipelines"
    PROJECT_CONTENT = "project_content"
    RL_SERVICE = "rl_service"
    RUN_HISTORY = "run_history"
    STUDIO = "studio"

    _names = []
    _discovery_keys = []
    _override_env_vars = []
    _name_to_key = {}
    _initialized = False

    @classmethod
    def initialize(cls):
        cls.register_discovery_url(cls.API, cls.API_DISCOVERY_KEY)
        cls.register_discovery_url(cls.ARTIFACTS, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.ASSETS, cls.MMS_DISCOVERY_KEY)
        cls.register_discovery_url(cls.CREDENTIAL, cls.EXPERIMENTATION_DISCOVERY_KEY)
        cls.register_discovery_url(cls.ENVIRONMENT, cls.EXPERIMENTATION_DISCOVERY_KEY)
        cls.register_discovery_url(cls.EXECUTION, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.EXPERIMENTATION, cls.EXPERIMENTATION_DISCOVERY_KEY)
        cls.register_discovery_url(cls.HYPERDRIVE, cls.HYPERDRIVE_DISCOVERY_KEY)
        cls.register_discovery_url(cls.JASMINE, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.METRICS, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.MLFLOW, cls.API_DISCOVERY_KEY)
        cls.register_discovery_url(cls.MODELMANAGEMENT, cls.MMS_DISCOVERY_KEY)
        cls.register_discovery_url(cls.PIPELINES, cls.API_DISCOVERY_KEY)
        cls.register_discovery_url(cls.PROJECT_CONTENT, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.RL_SERVICE, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.RUN_HISTORY, cls.HISTORY_DISCOVERY_KEY)
        cls.register_discovery_url(cls.STUDIO, cls.STUDIO_DISCOVERY_KEY)

        # For datastore:
        # cls.register_discover_url(cls.DATASTORE, cls.DATASTORE_DISCOVERY_KEY)
        cls._initialized = True

    @classmethod
    def register_discovery_url(cls, name, discovery_key=None):
        cls._names.append(name)
        discovery_key = discovery_key if discovery_key is not None else name
        cls._discovery_keys.append(discovery_key)

        cls._name_to_key[name] = discovery_key

        env_var = cls.generate_environment_variable(name)
        cls._override_env_vars.append(env_var)

    @classmethod
    def generate_environment_variable(cls, key):
        env_var = DEVLEOPER_URL_TEMPLATE.format(key.upper())
        return env_var

    @classmethod
    def get_discovery_key(cls, name):
        return cls._name_to_key[name]

    @classmethod
    def get_names(cls):
        return cls._names

    @classmethod
    def get_discovery_keys(cls):
        return cls._discovery_keys

    @classmethod
    def get_override_env_vars(cls):
        return cls._override_env_vars


_ServiceKeys.initialize()


class ServiceContext(ChainedIdentity):
    """
    A container for workspace scope, authentication, and service location information.

    :param subscription_id: The subscription id.
    :type subscription_id: str
    :param resource_group_name: The name of the resource group.
    :type resource_group_name: str
    :param workspace_name: The name of the workspace.
    :type workspace_name: str
    :param authentication: The auth object.
    :type authentication: azureml.core.authentication.AbstractAuthentication
    :param worker_pool:
    :type worker_pool: azureml._async.worker_pool.WorkerPool
    :param kwargs:
    """

    _worker_pool = None

    def __init__(self,
                 subscription_id, resource_group_name, workspace_name, workspace_id,
                 workspace_discovery_url, authentication, worker_pool=None, **kwargs):
        """
        Store scope information and fetch service locations.

        :param subscription_id: The subscription id.
        :type subscription_id: str
        :param resource_group_name: The name of the resource group.
        :type resource_group_name: str
        :param workspace_name: The name of the workspace.
        :type workspace_name: str
        :param workspace_discovery_url: The discovery url of the workspace.
        :type workspace_discovery_url: str
        :param authentication: The auth object.
        :type authentication: azureml.core.authentication.AbstractAuthentication
        :param kwargs:
        """
        super(ServiceContext, self).__init__(**kwargs)

        self._sub_id = subscription_id
        self._rg_name = resource_group_name
        self._ws_name = workspace_name
        self._workspace_id = workspace_id
        self._workspace_discovery_url = workspace_discovery_url
        self.worker_pool = worker_pool if worker_pool is not None else ServiceContext._get_worker_pool()

        from azureml.core.authentication import AbstractAuthentication
        assert isinstance(authentication, AbstractAuthentication)
        self._authentication = authentication

        self._endpoints = self._fetch_endpoints()

        self.runhistory_restclient = None
        self.armtemplate_restclient = None
        self.artifacts_restclient = None
        self.assets_restclient = None
        self.modelmanagement_restclient = None
        self.metrics_restclient = None
        self.project_content_restclient = None
        self.execution_restclient = None
        self.environment_restclient = None
        self.credential_restclient = None
        self.jasmine_restclient = None
        self._session = None

    def __deepcopy__(self, memo):
        # Details on how to override deepcopy https://stackoverflow.com/questions/1500718
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Skip serializing RestClient and WorkerPool as they will get re-initialized in the new object.
            if isinstance(v, RestClient) or isinstance(v, WorkerPool):
                setattr(result, k, None)
            # Skip serializing the logger and initialize it in the new object.
            elif isinstance(v, logging.Logger):
                setattr(result, k, logging.getLogger(v.name))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def subscription_id(self):
        """
        Return the subscription id for this workspace.

        :return: The subscription id.
        :rtype: str
        """
        return self._sub_id

    @property
    def resource_group_name(self):
        """
        Return the resource group name for this workspace.

        :return: The resource group name.
        :rtype: str
        """
        return self._rg_name

    @property
    def workspace_name(self):
        """
        Return the workspace name.

        :return: The workspace name.
        :rtype: str
        """
        return self._ws_name

    @property
    def workspace_id(self):
        """
        Return the workspace id.

        :return: The workspace id.
        :rtype: str
        """
        return self._workspace_id

    @classmethod
    def _get_worker_pool(cls):
        if cls._worker_pool is None:
            cls._worker_pool = WorkerPool(_parent_logger=module_logger)
            module_logger.debug("Created a static thread pool for {} class".format(cls.__name__))
        else:
            module_logger.debug("Access an existing static threadpool for {} class".format(cls.__name__))
        return cls._worker_pool

    def get_auth(self):
        """
        Return the authentication object.

        :return: The authentication object.
        :rtype: azureml.core.authentication.AbstractAuthentication
        """
        return self._authentication

    def _get_workspace_scope(self):
        """
        Return the scope information for the workspace.

        :param workspace_object: The workspace object.
        :type workspace_object: azureml.core.workspace.Workspace
        :return: The scope information for the workspace.
        :rtype: str
        """
        return ("/subscriptions/{}/resourceGroups/{}/providers"
                "/Microsoft.MachineLearningServices"
                "/workspaces/{}").format(self.subscription_id,
                                         self.resource_group_name,
                                         self.workspace_name)

    def _get_experiment_scope(self, experiment_name):
        return self._get_workspace_scope() + "/experiments/{}".format(experiment_name)

    def _get_developer_override(self, key):
        environment_variable = _ServiceKeys.generate_environment_variable(key)
        developer_endpoint = os.environ.get(environment_variable)
        if developer_endpoint is not None:
            self._logger.debug("Loaded endpoint {} from environment variable {}.".format(
                developer_endpoint, environment_variable))
        return developer_endpoint

    def _fetch_endpoints(self):
        """
        Fetch service endpoints from service discovery.

        :return: Dictionary of service discovery key to service url.
        :rtype: dict[str] -> str
        """
        scope = self._get_workspace_scope()
        endpoints = {}
        for discovery_key in _ServiceKeys.get_discovery_keys():
            url = get_service_url(self._authentication, scope, self._workspace_id,
                                  workspace_discovery_url=self._workspace_discovery_url,
                                  service_name=discovery_key)
            endpoints[discovery_key] = url
        return endpoints

    def _get_discovery_url(self):
        return self._workspace_discovery_url

    def _get_endpoint(self, key):
        """
        Get service endpoint from environment variable if set, otherwise from service discovery.

        :return: The service endpoint.
        :rtype: str
        """
        developer_endpoint = self._get_developer_override(key)
        if developer_endpoint is not None:
            endpoint = developer_endpoint
        else:
            discovery_key = _ServiceKeys.get_discovery_key(key)
            endpoint = self._endpoints[discovery_key]
        return endpoint

    def _get_api_url(self):
        """
        Get service endpoint for api

        :return: api endpoint
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.API)

    def _get_run_history_url(self):
        """
        Return the url to the run history service.

        :return: The run history service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.RUN_HISTORY)

    def _get_metrics_url(self):
        """
        Return the url to the metrics service.

        :return: The metrics service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.METRICS)

    def _get_jasmine_url(self):
        """
        Return the url to Jasmine service.

        :return: The Jasmine service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.JASMINE)

    def _get_experimentation_url(self):
        """
        Return the url to the experimentation service.

        :return: The experimentation service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.EXPERIMENTATION)

    def _get_artifacts_url(self):
        """
        Return the url to the artifacts service.

        :return: The artifacts service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.ARTIFACTS)

    def _get_pipelines_url(self):
        """
        Return the url to the pipelines service.

        :return: The pipelines service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.PIPELINES)

    def _get_hyperdrive_url(self):
        """
        Return the url to the hyperdrive service.

        :return: The hyperdrive service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.HYPERDRIVE)

    def _get_modelmanagement_url(self):
        """
        Return the url to the model management service.

        :return: The model management service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.MODELMANAGEMENT)

    def _get_assets_url(self):
        """
        Return the url to the assets service.

        :return: The assets service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.ASSETS)

    def _get_project_content_url(self):
        """
        Return the url to the project content service.

        :return: The project content service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.PROJECT_CONTENT)

    def _get_execution_url(self):
        """
        Return the url to the execution service.

        :return: The execution service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.EXECUTION)

    def _get_rl_service_url(self):
        """
        Return the url to the reinforcement learning service.

        :return: The reinforcement learning service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.RL_SERVICE)

    def _get_environment_url(self):
        """
        Return the url to the environment management service.

        :return: The environment management service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.ENVIRONMENT)

    def _get_credential_url(self):
        """
        Return the url to the credential service.

        :return: The credential service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.CREDENTIAL)

    def _get_workspace_portal_url(self):
        """
        Return the url to the workspace portal

        :return: The workspace portal endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.STUDIO)

    def _get_mlflow_url(self):
        """
        Return the url to the mlflow service

        :return: The mlflow service endpoint.
        :rtype: str
        """
        return self._get_endpoint(_ServiceKeys.MLFLOW)

    def _get_arm_template_restclient(self, host=None, user_agent=None):
        if self.armtemplate_restclient is None:
            host = host if host is not None else self._get_api_url()
            self.armtemplate_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.armtemplate_restclient, user_agent=user_agent)

        return self.armtemplate_restclient

    def _get_run_history_restclient(self, host=None, user_agent=None):
        if self.runhistory_restclient is None:
            host = host if host is not None else self._get_run_history_url()
            self.runhistory_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.runhistory_restclient, user_agent=user_agent)

        return self.runhistory_restclient

    def _get_metrics_restclient(self, user_agent=None):
        if self.metrics_restclient is None:
            self.metrics_restclient = RestClient(self.get_auth(), base_url=self._get_metrics_url())
            self._add_user_agent(self.metrics_restclient, user_agent=user_agent)

        return self.metrics_restclient

    def _get_jasmine_restclient(self, user_agent=None):
        if self.jasmine_restclient is None:
            self.jasmine_restclient = RestClient(self.get_auth(), base_url=self._get_jasmine_url())
            self._add_user_agent(self.jasmine_restclient, user_agent=user_agent)

        return self.jasmine_restclient

    def _get_artifacts_restclient(self, user_agent=None):
        if self.artifacts_restclient is None:
            self.artifacts_restclient = RestClient(self.get_auth(), base_url=self._get_artifacts_url())
            self._add_user_agent(self.artifacts_restclient, user_agent=user_agent)

        return self.artifacts_restclient

    def _get_modelmanagement_restclient(self, user_agent=None):
        if self.modelmanagement_restclient is None:
            self.modelmanagement_restclient = RestClient(self.get_auth(), base_url=self._get_modelmanagement_url())
            self._add_user_agent(self.modelmanagement_restclient, user_agent=user_agent)

        return self.modelmanagement_restclient

    def _get_assets_restclient(self, user_agent=None):
        if self.assets_restclient is None:
            self.assets_restclient = RestClient(self.get_auth(), base_url=self._get_assets_url())
            self._add_user_agent(self.assets_restclient, user_agent=user_agent)

        return self.assets_restclient

    def _get_project_content_restclient(self, user_agent=None):
        if self.project_content_restclient is None:
            host = self._get_project_content_url()
            self.project_content_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.project_content_restclient, user_agent=user_agent)

        return self.project_content_restclient

    def _get_execution_restclient(self, user_agent=None):
        if self.execution_restclient is None:
            host = self._get_execution_url()
            self.execution_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.execution_restclient, user_agent=user_agent)

        return self.execution_restclient

    def _get_environment_restclient(self, user_agent=None):
        if self.environment_restclient is None:
            host = self._get_environment_url()
            self.environment_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.environment_restclient, user_agent=user_agent)

        return self.environment_restclient

    def _get_credential_restclient(self, user_agent=None):
        if self.credential_restclient is None:
            host = self._get_credential_url()
            self.credential_restclient = RestClient(self.get_auth(), base_url=host)
            self._add_user_agent(self.credential_restclient, user_agent=user_agent)

        return self.credential_restclient

    def _get_shared_session(self, **kwargs):
        if self._session is None:
            self._session = requests.session()
            self._session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3, **kwargs))
        return self._session

    def _add_user_agent(self, rest_client, user_agent=None):
        if user_agent is not None:
            if isinstance(user_agent, six.text_type):
                rest_client.config.add_user_agent(user_agent)
            else:
                self._logger.warning("Client agent setting is not a string, it is ignored.")
