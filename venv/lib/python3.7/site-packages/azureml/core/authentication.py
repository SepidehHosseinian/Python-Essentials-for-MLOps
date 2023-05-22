# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing different types of authentication in Azure Machine Learning.

Types of supported authentication:

* Interactive Login - The default mode when using Azure Machine Learning SDK. Uses an interactive dialog.
* Azure CLI - For use with the [azure-cli](https://docs.microsoft.com/cli/azure) package.
* Service Principal - For use with automated machine learning workflows.
* MSI - For use with Managed Service Identity-enabled assets such as with an Azure Virtual Machine.
* Azure ML Token - Used for acquiring Azure ML tokens for submitted runs only.

To learn more about these authentication mechanisms, see https://aka.ms/aml-notebook-auth.
"""
import datetime
import errno
import logging
import os
import pytz
import time
import threading
import jwt
import re
import dateutil.parser
import base64
import hashlib
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from pkg_resources import parse_version

from azureml._common.exceptions import AzureMLException
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from abc import ABCMeta, abstractmethod
from six import raise_from

import collections
from requests import Session
from inspect import getfullargspec

from azureml.core.util import NewLoggingLevel
from azureml.exceptions import AuthenticationException, RunEnvironmentException, UserErrorException
from azureml._async.daemon import Daemon
from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token, perform_interactive_login
from azureml._base_sdk_common.service_discovery import DISCOVERY_END_POINT
from azureml._restclient.clientbase import execute_func
from azureml._restclient.service_context import ServiceContext
from azureml._logging.logged_lock import ACQUIRE_DURATION_THRESHOLD, LoggedLock
from azureml._vendor.azure_cli_core.auth.util import resource_to_scopes

_SubscriptionInfo = collections.namedtuple("SubscriptionInfo", "subscription_name subscription_id")

_TOKEN_REFRESH_THRESHOLD_SEC = 5 * 60

_AZUREML_SERVICE_PRINCIPAL_TENANT_ID_ENV_VAR = "AZUREML_SERVICE_PRINCIPAL_TENANT_ID"
_AZUREML_SERVICE_PRINCIPAL_ID_ENV_VAR = "AZUREML_SERVICE_PRINCIPAL_ID"
_AZUREML_SERVICE_PRINCIPAL_PASSWORD_ENV_VAR = "AZUREML_SERVICE_PRINCIPAL_PASSWORD"
_TENANT_ID = 'tenantId'

module_logger = logging.getLogger(__name__)

_AML_RESOURCE_ID_DICT = {
    "AzureCloud": "https://ml.azure.com",
    "AzureChinaCloud": "https://ml.azure.cn",
    "AzureUSGovernment": "https://ml.azure.us",
}

_AML_RESOURCE_ID_PREFIX = "https://ml.azure"

if TYPE_CHECKING:
    from typing import NamedTuple
    AccessToken = NamedTuple("AccessToken", [("token", str), ("expires_on", int)])
else:
    from collections import namedtuple
    AccessToken = namedtuple("AccessToken", ["token", "expires_on"])


class Audience(Enum):
    """Audience supported by AML. To be used only with `TokenAuthentication` class."""

    ARM = "ARM"
    AZUREML = "AZUREML"


class AbstractAuthentication(object):
    """Abstract parent class for all authentication classes in Azure Machine Learning.

    Derived classes provide different means to authenticate and acquire a token based on their targeted use case.
    For examples of authentication, see https://aka.ms/aml-notebook-auth.

    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
        If no default is found, "AzureCloud" is used.
    :type cloud: str
    """

    __metaclass__ = ABCMeta

    def __init__(self, cloud=None):
        """Class AbstractAuthentication constructor.

        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
            If no default is found, "AzureCloud" is used.
        :type cloud: str
        """
        from azureml._vendor.azure_cli_core.azureml_cloud import _Clouds
        self._cloud_type = _Clouds.get_cloud_or_default(cloud)

    def get_authentication_header(self):
        """Return the HTTP authorization header.

        The authorization header contains the user access token for access authorization against the service.

        :return: Returns the HTTP authorization header.

        :rtype: dict
        """
        # We return a new dictionary each time, as some functions modify the headers returned
        # by this function.
        auth_header = {"Authorization": "Bearer " + self._get_arm_token()}
        return auth_header

    def get_token(self, *scopes, **kwargs):
        """Contract for Track 2 SDKs to get token.

        Currently supports Auth classes with self.get_authentication_header function implemented.

        :param scopes: Args.
        :param kwargs: Kwargs.
        :return: Returns a named tuple.
        :rtype: collections.namedtuple
        """
        # TODO: This is being implemented to move away from Old Adal Auth class that just holds tokens
        token = self.get_authentication_header()["Authorization"].split(" ")[1]
        return AccessToken(token, int(_get_exp_time(token)))

    def _get_azureml_client_authentication_header(self):
        """Return the HTTP authorization header.

        The authorization header contain the azureml user access token for access authorization against the service.

        :return: Returns the HTTP authorization header.

        :rtype: dict
        """
        # We return a new dictionary each time, as some functions modify the headers returned
        # by this function.
        auth_header = {"Authorization": "Bearer " + self._get_azureml_client_token()}
        return auth_header

    @abstractmethod
    def _get_arm_token(self):
        """Abstract method that auth classes should implement to return an arm token.

        :return: Return a user's arm token.
        :rtype: str
        """
        pass

    @abstractmethod
    def _get_graph_token(self):
        """Abstract method that auth classes should implement to return a Graph token.

        :return: Return a user's Graph token.
        :rtype: str
        """
        pass

    @abstractmethod
    def _get_azureml_client_token(self):
        """Abstract method that auth classes should implement to return an AzureML client token.

        :return: Return a user's AzureML client token.
        :rtype: str
        """
        pass

    @abstractmethod
    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        :return: Returns a list of SubscriptionInfo named tuples, and the current tenant id.
        :rtype: list, str
        """
        pass

    @abstractmethod
    def _get_workspace(self, subscription_id, resource_group, name):
        """Return a workspace when the auth object can't talk to ARM.

        :return: A Workspace matching the specified subscription ID, resource group, and name,
            or None to fallback to using ARM.
        :rtype: azureml.core.Workspace or None
        """
        pass

    def _get_service_client(self, client_class, subscription_id, subscription_bound=True, base_url=None, is_check_subscription=True):
        """Create and return a service client object for client_class using subscription_id and auth token.

        :param client_class:
        :type client_class: object
        :param subscription_id:
        :type subscription_id: str
        :param subscription_bound: True if a subscription id is required to construct the client. Only False for
        1 RP clients.
        :type subscription_bound: bool
        :param base_url: The specified base URL client should use, usually differs by region
        :type base_url: str
        :param is_check_subscription: Check if subscription exists or not.
        :type is_check_subscription: bool
        :return:
        :rtype: client_class
        """
        # Checks if auth has access to the provided subscription.
        # In Azureml Token based auth, we don't do subscription check, as this requires querying ARM.
        # We don't use az CLI methods to get a service client because in multi-tenant case, based on a subscription
        # az CLI code changes the arm token while getting a service client, which means that
        # the arm token that this auth object has differs from the arm token in the service client
        # in the multi-tenant case, which causes confusion.
        if subscription_id and is_check_subscription:
            all_subscription_list, tenant_id = self._get_all_subscription_ids()
            self._check_if_subscription_exists(subscription_id, all_subscription_list, tenant_id)

        if not base_url:
            base_url = self._cloud_type.endpoints.resource_manager

        return _get_service_client_using_arm_token(self, client_class, subscription_id,
                                                   subscription_bound=subscription_bound,
                                                   base_url=base_url)

    def signed_session(self, session=None):
        """Add the authorization header as a persisted header on an HTTP session.

        Any new requests sent by the session will contain the authorization header.

        :param session: The HTTP session that will have the authorization header as a default persisted header.
            When None, a new session is created.
        :type session: requests.sessions.Session
        :return: Returns the HTTP session after the update.
        :rtype: requests.sessions.Session
        """
        session = session or Session()
        session.headers.update(self.get_authentication_header())
        return session

    def _get_aml_resource_id(self, cloud=None):
        """Return aml resource id.

        :param cloud: cloud for which aml resource id will be returned
        :type cloud: str
        :return: Returns aml resource id
        :type: str
        """
        suffix = self._get_cloud_suffix(cloud=cloud)
        return ".".join([_AML_RESOURCE_ID_PREFIX, suffix])

    def _get_cloud_suffix(self, cloud=None):
        cloud = cloud if cloud else self._cloud_type.name
        from azureml._vendor.azure_cli_core.azureml_cloud import _Clouds
        cloud_metadata = _Clouds.get_cloud_or_default(cloud_name=cloud)
        try:
            return cloud_metadata.endpoints.active_directorys.split(".")[-1]
        except Exception as e:
            logging.debug("Failed to get cloud suffix")

        if cloud not in _AML_RESOURCE_ID_DICT.keys():
            raise ValueError("Either cloud is incorrect or AML is not supported in the cloud")
        return _AML_RESOURCE_ID_DICT.get(cloud).split(".")[-1]

    def _get_arm_end_point(self):
        """Return the arm end point.

        :return: Returns the arm end point.
        :rtype: str
        """
        return self._cloud_type.endpoints.resource_manager

    def _get_cloud_type(self):
        """Return the cloud type.

        :return:
        :rtype: azureml._vendor.azure_cli_core.cloud.Cloud
        """
        return self._cloud_type

    def _get_adal_auth_object(self, is_graph_auth=False):
        """Return an adal auth object.

        The old_adal_authentication has a class just to hold tokens inside an anonymous function.
        It in no way connects with adal python package. No "import adal" statements.
        We are using MSAL.

        :return: Returns adal auth object needed for azure sdk clients.
        :rtype: azureml._vendor.azure_cli_core.auth.old_adal_authentication.AdalAuthentication
        """
        from azureml._vendor.azure_cli_core.auth.old_adal_authentication import AdalAuthentication
        if is_graph_auth:
            token = self._get_graph_token()
        else:
            token = self.get_authentication_header()["Authorization"].split(" ")[1]
        token_expiry = {"expires_on": int(_get_exp_time(token))}
        adal_auth_object = AdalAuthentication(lambda x: ("Bearer", token, token_expiry))
        return adal_auth_object

    def _check_if_subscription_exists(self, subscription_id, subscription_id_list, tenant_id):
        """Check if subscription_id exists in subscription_id_list.

        :param subscription_id: Subscription id to check.
        :type subscription_id: str
        :param subscription_id_list: Subscription id list.
        :type subscription_id_list: list[azureml.core.authentication.SubscriptionInfo]
        :param tenant_id: Currently logged-in tenant id
        :type tenant_id: str
        :return: True if subscription exists.
        :rtype: bool
        """
        name_matched = False
        for subscription_info in subscription_id_list:
            if subscription_info.subscription_id.lower().strip() == subscription_id.lower().strip():
                return True
            if subscription_info.subscription_name.lower().strip() == subscription_id.lower().strip():
                name_matched = True

        if name_matched:
            raise UserErrorException("It looks like you have specified subscription name, {}, instead of "
                                     "subscription id. Subscription names may not be unique, please specify "
                                     "subscription id from this list \n {}".format(subscription_id,
                                                                                   subscription_id_list))
        else:
            raise UserErrorException("You are currently logged-in to {} tenant. You don't have access "
                                     "to {} subscription, please check if it is in this tenant. "
                                     "All the subscriptions that you have access to in this tenant are = \n "
                                     "{}. \n Please refer to aka.ms/aml-notebook-auth for different "
                                     "authentication mechanisms in azureml-sdk.".format(tenant_id,
                                                                                        subscription_id,
                                                                                        subscription_id_list))

def _login_on_failure_decorator(lock_to_use):
    """Login on failure decorator.

    This decorator performs az login on failure of the actual function and retries the actual function one more
    time.

    Notebooks are long running processes, like people open them and then never close them.
    So, on InteractiveLoginAuthentication object, auth._get_arm_token etc functions can throw
    error if the arm token and refresh token both have expired, and this will prompt a
    user to run "az login" outside the notebook. So, we use this decorator on every function
    of InteractiveLoginAuthentication to catch the first failure, perform "az login" and retry the
    function.

    We also use Profile(async_persist=False), so that tokens are persisted and re-used on-disk.
    async_persist=True (default), az CLI only persists tokens on the process exit, which just never
    happens in a notebook case, in turn requiring users to perform "az login" outside notebooks.

    :return:
    :rtype: object
    """
    def actual_decorator(test_function):
        """Actual decorator.

        :param test_function:
        :type test_function: object
        :return: Returns the wrapper.
        :rtype: object
        """
        def wrapper(self, *args, **kwargs):
            """Wrapper.

            :param args:
            :type args: list
            :param kwargs:
            :type kwargs: dict
            :return: Returns the test function.
            :rtype: object
            """
            try:
                start_time = time.time()
                lock_to_use.acquire()
                duration = time.time() - start_time
                if duration > ACQUIRE_DURATION_THRESHOLD:
                    module_logger.debug("{} acquired lock in {} s.".format(type(self).__name__, duration))
                return test_function(self, *args, **kwargs)
            except Exception as e:
                if type(self) == InteractiveLoginAuthentication:
                    # Perform InteractiveLoginAuthentication and try one more time.
                    InteractiveLoginAuthentication(force=True, tenant_id=self._tenant_id)
                    # Try one more time
                    return test_function(self, *args, **kwargs)
                else:
                    raise e
            finally:
                lock_to_use.release()
        return wrapper

    return actual_decorator


def _retry_connection_aborted(*, retries):
    """Retry method on connection aborted error.

    :param retries: Maximum number of retries.
    :type retries: int
    :return: Decorator.
    :rtype: object
    """
    def actual_decorator(function):
        """Actual decorator.

        :param function: Callable to be wrapped.
        :type function: object
        :return: Wrapper around function.
        :rtype: object
        """
        def connection_aborted_wrapper(*args, **kwargs):
            """Execute the wrapped function, retrying as needed."""
            attempt = 0
            while True:
                try:
                    return function(*args, **kwargs)
                except AuthenticationException as e:
                    if "Connection aborted." in str(e) and attempt <= retries:
                        module_logger.debug("Caught connection aborted exception on attempt {} of {}:\n{}"
                                            .format(attempt, retries, e))
                        attempt += 1
                        continue
                    raise

        return connection_aborted_wrapper

    return actual_decorator


class InteractiveLoginAuthentication(AbstractAuthentication):
    """Manages authentication and acquires an authorization token in interactive login workflows.

    Interactive login authentication is suitable for local experimentation on your own computer, and is the
    default authentication model when using Azure Machine Learning SDK. For example, when working locally in
    a Jupyter notebook, the interactive login authentication process opens a browser window opens to prompt for
    credentials if credentials don't already exist.

    .. remarks::

        The constructor of the class will prompt you to login. The constructor then will save the credentials
        for any subsequent attempts. If you are already logged in with the Azure CLI or have logged-in before, the
        constructor will load the existing credentials without prompt.

        .. code-block:: python

            from azureml.core.authentication import InteractiveLoginAuthentication

            interactive_auth = InteractiveLoginAuthentication()
            auth_header = interactive_auth.get_authentication_header()
            print(auth_header)

        You can also initiate an interactive loging using the :meth:`azureml.core.Workspace.from_config` method
        of the :class:`azureml.core.Workspace` class.

        When this Python process is running in Azure Notebook service, the constructor will attempt to use the
        "connect to azure" feature in Azure Notebooks.

        If this Python process is running on a Notebook VM, the constructor will attempt to use MSI authentication.

        In some use cases you may need to specify a tenant ID. For example, when you are accessing a subscription
        as a guest to a tenant that is not your default, you will need to specify the tenant ID of the
        Azure Active Directory you're using as shown in the following example.

        .. code-block:: python

            from azureml.core.authentication import InteractiveLoginAuthentication

            interactive_auth = InteractiveLoginAuthentication(tenant_id="my-tenant-id")

            ws = Workspace(subscription_id="my-subscription-id",
                           resource_group="my-ml-rg",
                           workspace_name="my-ml-workspace",
                           auth=interactive_auth)

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb


    :param force: Indicates whether "az login" will be run even if the old "az login" is still valid.
        The default is False.
    :type force: bool
    :param tenant_id: The tenant ID to login in to. This is can be used to specify a specific tenant when
        you have access to multiple tenants. If unspecified, the default tenant will be used.
    :type tenant_id: str
    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI
        is used. If no default is found, "AzureCloud" is used.
    :type cloud: str
    """

    # We are using locks because in some cases like hyperdrive this
    # function gets called in parallel, which initiates multiple
    # "az login" at once.
    # TODO: This needs to be made non-static.
    # Currently we are not sure if it is a parallelism issue or multiple auth objects.
    # So, going with static global lock.
    _interactive_auth_lock = threading.Lock()

    # TODO: This authentication mechanism should use the azureml application ID while
    # authenticating with ARM, rather than the az CLI application ID.

    # TODO: This should also persist state separately from .azure directory, so we need to
    # implement the state saving mechanism in a thread-safe manner using locking.

    def __init__(self, force=False, tenant_id=None, cloud=None):
        """Class Interactive Login Authentication constructor.

        This constructor will prompt the user to login, then it will save the credentials for any subsequent
        attempts. If the user is already logged in to azure CLI  or have logged in before, the constructor will load
        the existing credentials without prompt. When this python process is running in Azure Notebook service, the
        constructor will attempt to use the "connect to azure" feature in Azure Notebooks.
        If this python process is running on a Notebook VM, the constructor will attempt to use MSI auth.

        :param force: Indicates whether "az login" will be run even if the old "az login" is still valid.
            The default is False.
        :type force: bool
        :param tenant_id: The tenant ID to login in to. This is can be used to specify a specific tenant when
            you have access to multiple tenants. If unspecified, the default tenant will be used.
        :type tenant_id: str
        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI
            is used. If no default is found, "AzureCloud" is used.
        :type cloud: str

        """
        # TODO: This is just based on az login for now.
        # This all will need to change and made az CLI independent.

        super(InteractiveLoginAuthentication, self).__init__(cloud)
        self._tenant_id = tenant_id

        arg_spec = getfullargspec(self._get_ambient)
        if len(arg_spec.args) == 1:
            # This is temporary solution until the bug is actually fixed in hosttools that overrides
            # self._get_ambient function.
            # Until this bug is fixed https://msdata.visualstudio.com/Vienna/_workitems/edit/1015204
            self._ambient_auth = self._get_ambient()
        else:
            self._ambient_auth = self._get_ambient(cloud)

        if self._ambient_auth is None:
            if force:
                print("Performing interactive authentication. Please follow the instructions "
                      "on the terminal.")
                perform_interactive_login(tenant=tenant_id, cloud_type=self._cloud_type)
                print("Interactive authentication successfully completed.")
            else:
                need_to_login = False
                try:
                    self._get_arm_token_using_interactive_auth()
                except Exception:
                    try:
                        self._fallback_to_azure_cli_credential()
                    except Exception:
                        need_to_login = True

                if need_to_login:
                    print("Performing interactive authentication. Please follow the instructions "
                          "on the terminal.")
                    perform_interactive_login(tenant=tenant_id, cloud_type=self._cloud_type)
                    print("Interactive authentication successfully completed.")

    @_login_on_failure_decorator(_interactive_auth_lock)
    def _get_resource_token(self, resource):
        """Return the access token of resource.

        :return: The access token of resource.
        :rtype: str
        """
        if isinstance(self._ambient_auth, AbstractAuthentication):
            if hasattr(self._ambient_auth, '_get_resource_token'):
                return self._ambient_auth._get_resource_token(resource)
            raise NotImplementedError(
                "No token for ambient authentication type {}".format(type(self._ambient_auth).__name__))
        else:
            return self._get_arm_token_using_interactive_auth(resource=resource)

    @_login_on_failure_decorator(_interactive_auth_lock)
    def _get_arm_token(self):
        """Return the arm access token.

        :return: Returns the arm access token.
        :rtype: str
        """
        if isinstance(self._ambient_auth, AbstractAuthentication):
            return self._ambient_auth._get_arm_token()
        else:
            return self._get_arm_token_using_interactive_auth()

    @_login_on_failure_decorator(_interactive_auth_lock)
    def _get_graph_token(self):
        """Return the Graph access token.

        :return: Returns the Graph access token.
        :rtype: str
        """
        if isinstance(self._ambient_auth, AbstractAuthentication):
            if hasattr(self._ambient_auth, '_get_graph_token'):
                return self._ambient_auth._get_graph_token()
            raise NotImplementedError(
                "No graph token for ambient authentication type {}".format(type(self._ambient_auth).__name__))
        else:
            return self._get_arm_token_using_interactive_auth(
                resource=self._cloud_type.endpoints.active_directory_graph_resource_id)

    @_login_on_failure_decorator(_interactive_auth_lock)
    def _get_azureml_client_token(self):
        """Return the Graph access token.

        :return: Returns the Graph access token.
        :rtype: str
        """
        if isinstance(self._ambient_auth, AbstractAuthentication):
            if hasattr(self._ambient_auth, '_get_azureml_client_token'):
                return self._ambient_auth._get_azureml_client_token()
            raise NotImplementedError(
                "No azureml client token for ambient authentication type {}".format(type(self._ambient_auth).__name__))
        else:
            return self._get_arm_token_using_interactive_auth(
                resource=self._get_aml_resource_id())

    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        :return: Returns a list of SubscriptionInfo named tuples.
        :rtype: list, str
        """
        arm_token = self._get_arm_token()
        return self._get_all_subscription_ids_internal(arm_token)

    def _get_workspace(self, subscription_id, resource_group, name):
        """Return a workspace when the auth object can't talk to ARM.

        :return: A Workspace matching the specified subscription ID, resource group, and name,
            or None to fallback to using ARM.
        :rtype: azureml.core.Workspace or None
        """
        if isinstance(self._ambient_auth, AbstractAuthentication):
            return self._ambient_auth._get_workspace(subscription_id, resource_group, name)
        else:
            return None

    @_login_on_failure_decorator(_interactive_auth_lock)
    def _get_all_subscription_ids_internal(self, arm_token):
        if isinstance(self._ambient_auth, AbstractAuthentication):
            return self._ambient_auth._get_all_subscription_ids()
        else:
            from azureml._vendor.azure_cli_core._profile import Profile
            from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token
            token_tenant_id = fetch_tenantid_from_aad_token(arm_token)
            profile_object = _get_profile(self._cloud_type)
            # List of subscriptions might be cached on disk, so we are just trying to get it from disk.
            all_subscriptions = profile_object.load_cached_subscriptions()
            result = []

            for subscription_info in all_subscriptions:
                # Az CLI returns subscriptions for all tenants, but we want subscriptions
                # that a user has access to using the current arm token for the tenant. So we
                # filter based on tenant id.
                # There are some subscriptions from windows azure time that don't have
                # tenant id for them, so we ignore tenantId check for them.
                if "tenantId" not in subscription_info or \
                        subscription_info["tenantId"].lower() == token_tenant_id.lower():
                    subscription_tuple = _SubscriptionInfo(subscription_info['name'],
                                                           subscription_info['id'])
                    result.append(subscription_tuple)
            return result, token_tenant_id

    def _check_if_subscription_exists(self, subscription_id, subscription_id_list, tenant_id):
        super(InteractiveLoginAuthentication, self)._check_if_subscription_exists(subscription_id,
                                                                                  subscription_id_list, tenant_id)

    def _get_ambient(self, cloud):
        ambient_auth = None

        if not ambient_auth:
            # try MSI
            ambient_auth = MsiAuthentication._initialize_msi_auth(tenant_id_to_check=self._tenant_id, cloud=cloud)

        if not ambient_auth:
            # try SP from env vars
            ambient_auth = ServicePrincipalAuthentication._initialize_sp_auth(self._tenant_id, cloud=cloud)

        if not ambient_auth:
            # For arcadia environment
            ambient_auth = _ArcadiaAuthentication._initialize_arcadia_auth(tenant_id_to_check=self._tenant_id, cloud=cloud)

        if not ambient_auth:
            # Check if we're in a high-concurrency ADB cluster
            ambient_auth = _DatabricksClusterAuthentication._initialize_adb_auth(tenant_id_to_check=self._tenant_id)

        if not ambient_auth:
            # Use an ARM token from the workspace key vault.
            ambient_auth = ArmTokenAuthentication._initialize_arm_token_auth(self._tenant_id, cloud=cloud)

        if not ambient_auth:
            # Use the Azure ML run token.
            ambient_auth = AzureMLTokenAuthentication._initialize_aml_token_auth(self._tenant_id)

        return ambient_auth

    def _get_arm_token_using_interactive_auth(self, force_reload=False, resource=None):
        """Get the arm token cached on disk in interactive auth.

        :param force_reload: Force reloads credential information from disk.
        :type force_reload: bool
        :return: arm token or exception.
        :rtype: str
        """
        from azureml._vendor.azure_cli_core._session import ACCOUNT, CONFIG, SESSION
        from azureml._vendor.azure_cli_core._environment import get_config_dir
        profile_object = _get_profile(self._cloud_type)

        # If we can get a valid ARM token, then we don't need to login.
        arm_token = _get_arm_token_with_refresh(
            profile_object, ACCOUNT, CONFIG, SESSION, get_config_dir(),
            force_reload=force_reload, resource=resource
        )
        # If a user has specified a tenant id then we need to check if this token is for that tenant.
        if self._tenant_id and fetch_tenantid_from_aad_token(arm_token) != self._tenant_id:
            raise UserErrorException("The tenant in the authentication is different "
                                     "than the user specified tenant.")
        else:
            return arm_token

    def _fallback_to_azure_cli_credential(self):
        from azureml._vendor.azure_cli_core._environment import _AZUREML_AUTH_CONFIG_DIR_ENV_NAME

        auth_config_dir_value = os.getenv(_AZUREML_AUTH_CONFIG_DIR_ENV_NAME, None)
        try:
            # Setting the environment variable for ~/.azure directory.
            os.environ[_AZUREML_AUTH_CONFIG_DIR_ENV_NAME] \
                = os.getenv('AZURE_CONFIG_DIR', None) or os.path.expanduser(os.path.join('~', '.azure'))
            # Resetting global credential cache.
            # Profile._global_creds_cache = None
            # Reloading the state from disk with new directory.
            self._get_arm_token_using_interactive_auth(force_reload=True)
            # If this succeeds then we keep using the directory, and even child process will inherit this
            # env variable.
            import logging
            logging.getLogger().warning("Warning: Falling back to use azure cli login credentials.\n"
                                        "If you run your code in unattended mode, i.e., where you can't give a user input, "
                                        "then we recommend to use ServicePrincipalAuthentication or MsiAuthentication.\n"
                                        "Please refer to aka.ms/aml-notebook-auth for different authentication mechanisms "
                                        "in azureml-sdk.")
        except Exception as ex:
            # Profile._global_creds_cache = None
            if auth_config_dir_value:
                os.environ[_AZUREML_AUTH_CONFIG_DIR_ENV_NAME] = auth_config_value
            else:
                # If this fails then we remove this env variable to use ~/.azureml/auth directory.
                del os.environ[_AZUREML_AUTH_CONFIG_DIR_ENV_NAME]
            raise ex


class AzureCliAuthentication(AbstractAuthentication):
    """Manages authentication and acquires an access token using the Azure CLI.

    To use this class you must have the **azure-cli** package installed. For a better Azure Notebooks
    experience, use the :class:`azureml.core.authentication.InteractiveLoginAuthentication` class.

    .. remarks::

        If you have installed **azure-cli** package, and used az login command to log in to your
        Azure Subscription, then you can use the AzureCliAuthentication class.


        .. code-block:: python

            from azureml.core.authentication import AzureCliAuthentication

            cli_auth = AzureCliAuthentication()

            ws = Workspace(subscription_id="my-subscription-id",
                           resource_group="my-ml-rg",
                           workspace_name="my-ml-workspace",
                           auth=cli_auth)

            print("Found workspace {} at location {}".format(ws.name, ws.location))

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb


    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is
        used. If no default is found, "AzureCloud" is used.
    :type cloud: str
    """

    def __init__(self, cloud=None):
        """Class Azure Cli Authentication constructor.

        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is
            used. If no default is found, "AzureCloud" is used.
        :type cloud: str
        """
        super(AzureCliAuthentication, self).__init__(cloud)
        self._azure_cli_core_check()

    _azcli_auth_lock = threading.Lock()

    _TOKEN_TIMEOUT_SECs = 3600

    @_login_on_failure_decorator(_azcli_auth_lock)
    def _get_default_subscription_id(self):
        """Return default subscription id.

        This method has lock, as it access az CLI internal methods.

        :return: Returns the default subscription id.
        :rtype: str
        """
        self._azure_cli_core_check()

        # Hack to make this work outside the Azure CLI.
        from azure.cli.core._session import ACCOUNT, CONFIG, SESSION
        from azure.cli.core._environment import get_config_dir
        from azure.cli.core.commands.client_factory import get_subscription_id
        from azure.cli.core import get_default_cli

        if not ACCOUNT.data:
            config_dir = get_config_dir()
            ACCOUNT.load(os.path.join(config_dir, 'azureProfile.json'))
            CONFIG.load(os.path.join(config_dir, 'az.json'))
            SESSION.load(os.path.join(config_dir, 'az.sess'), max_age=AzureCliAuthentication._TOKEN_TIMEOUT_SECs)

        return get_subscription_id(get_default_cli())

    @_login_on_failure_decorator(_azcli_auth_lock)
    def _get_arm_token(self):
        """Fetch a valid ARM token using azure API calls.

        This method has lock, as it access az CLI internal methods.

        :return: Arm access token.
        :rtype: str
        """
        self._azure_cli_core_check()

        from azure.cli.core._profile import Profile

        # Hack to make this work outside the Azure CLI.
        from azure.cli.core._session import ACCOUNT, CONFIG, SESSION
        from azure.cli.core._environment import get_config_dir

        # From notebook, we want to persist tokens synchronously to the on-disk file,
        # so that users don't have to run az login multiple times.
        # By default, a tokens are persisted on-disk on exit on a process, which
        # doesn't happen in notebook or happens in a way where token persistence logic
        # is not called.
        profile_object = Profile()

        return _get_arm_token_with_refresh(
            profile_object, ACCOUNT, CONFIG, SESSION, get_config_dir()
        )

    @_login_on_failure_decorator(_azcli_auth_lock)
    def _get_graph_token(self):
        """Fetch a valid Graph token using azure API calls.

        This method has lock, as it access az CLI internal methods.

        :return: Graph access token.
        :rtype: str
        """
        self._azure_cli_core_check()

        from azure.cli.core._profile import Profile

        # Hack to make this work outside the Azure CLI.
        from azure.cli.core._session import ACCOUNT, CONFIG, SESSION
        from azure.cli.core._environment import get_config_dir

        # From notebook, we want to persist tokens synchronously to the on-disk file,
        # so that users don't have to run az login multiple times.
        # By default, a tokens are persisted on-disk on exit on a process, which
        # doesn't happen in notebook or happens in a way where token persistence logic
        # is not called.
        profile_object = Profile()

        return _get_arm_token_with_refresh(
            profile_object,
            ACCOUNT,
            CONFIG,
            SESSION,
            get_config_dir(),
            resource=self._cloud_type.endpoints.active_directory_graph_resource_id
        )

    @_login_on_failure_decorator(_azcli_auth_lock)
    def _get_azureml_client_token(self):
        """Fetch a valid Graph token using azure API calls.

        This method has lock, as it access az CLI internal methods.

        :return: Graph access token.
        :rtype: str
        """
        self._azure_cli_core_check()

        from azure.cli.core._profile import Profile

        # Hack to make this work outside the Azure CLI.
        from azure.cli.core._session import ACCOUNT, CONFIG, SESSION
        from azure.cli.core._environment import get_config_dir

        # From notebook, we want to persist tokens synchronously to the on-disk file,
        # so that users don't have to run az login multiple times.
        # By default, a tokens are persisted on-disk on exit on a process, which
        # doesn't happen in notebook or happens in a way where token persistence logic
        # is not called.
        profile_object = Profile()

        return _get_arm_token_with_refresh(
            profile_object,
            ACCOUNT,
            CONFIG,
            SESSION,
            get_config_dir(),
            resource=self._get_aml_resource_id()
        )

    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        This method has lock, as it access az CLI internal methods.

        :return: Returns a list of SubscriptionInfo named tuples.
        :rtype: list, str
        """
        arm_token = self._get_arm_token()
        return self._get_all_subscription_ids_internal(arm_token)

    @_login_on_failure_decorator(_azcli_auth_lock)
    def _get_all_subscription_ids_internal(self, arm_token):
        self._azure_cli_core_check()

        from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token
        from azure.cli.core._profile import Profile
        profile = Profile()
        # This is a CLI authentication, so we are just calling simple CLI methods
        # for getting subscriptions.
        all_subscriptions = profile.load_cached_subscriptions()
        result = []

        token_tenant_id = fetch_tenantid_from_aad_token(arm_token)

        for subscription_info in all_subscriptions:

            # Az CLI returns subscriptions for all tenants, but we want subscriptions
            # that a user has access to using the current arm token for the tenant. So we
            # filter based on tenant id.
            # There are some subscriptions from windows azure time that don't have
            # tenant id for them, so we ignore tenantId check for them.
            if "tenantId" not in subscription_info or \
                    subscription_info["tenantId"].lower() == token_tenant_id.lower():
                subscription_tuple = _SubscriptionInfo(subscription_info['name'],
                                                       subscription_info['id'])
                result.append(subscription_tuple)
        return result, token_tenant_id

    @_login_on_failure_decorator(_azcli_auth_lock)
    def _get_refresh_token(self, resource=None):
        """Get token refresh credentials from the az CLI.

        :param resource:
        :type resource: str
        :return: User type and refresh credentials.
        :rtype: (str, (str, str, str, str))
        """
        from azure.cli.core._profile import Profile
        # This function is only used by notebook validation pipeline
        # submit-notebook (private) command calls this function
        try:
            profile = Profile()

            account = profile.get_subscription()
            # Set "Access service principal details in script" in your pipeline's task
            sp_id = os.environ['servicePrincipalId']
            sp_pass = os.environ['servicePrincipalKey']
            tenant_id = str(account[_TENANT_ID])
            return account['user']['type'], (sp_id, sp_pass, None, tenant_id)
        except KeyError as e:
            raise AuthenticationException(
                "Please add servicePrincipalId and servicePrincipalKey as environment variables. If"
                " you are running in Azure CLI task, check for 'Access service principal details in script'",
                inner_exception=e
            )
        except Exception as e:
            raise AuthenticationException("Could not retrieve user refresh token. Please run 'az login'.",
                                          inner_exception=e)

    @staticmethod
    def _azure_cli_core_check():
        try:
            import azure.cli.core # noqa
            from azure.cli.core._profile import Profile  # noqa
            from azure.cli.core.cloud import get_active_cloud  # noqa

            # Hack to make this work outside the Azure CLI.
            from azure.cli.core._session import ACCOUNT, CONFIG, SESSION  # noqa
            from azure.cli.core import get_default_cli  # noqa
            from azure.cli.core._environment import get_config_dir # noqa

            from azure.cli.core import __version__ as ver # noqa
        except Exception:
            raise AuthenticationException("azure-cli package is not installed. "
                                          "AzureCliAuthentication requires azure-cli>=2.30.0 to be installed "
                                          "in the same python environment where azureml-sdk is installed.")
        if parse_version('2.30.0') > parse_version(ver):
            raise AuthenticationException("AzureCliAuthentication requires azure-cli>=2.30.0 to be installed "
                                          "in the same python environment where azureml-sdk is installed.")


class ArmTokenAuthentication(AbstractAuthentication):
    """Used internally to acquire ARM access tokens using service principle or managed service identity authentication.

    For automated workflows where managed access control is needed, use the
    :class:`azureml.core.authentication.ServicePrincipalAuthentication` instead.

    :param arm_access_token: An ARM access token.
    :type arm_access_token: str
    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
        If no default is found, "AzureCloud" is used.
    :type cloud: str
    """

    def __init__(self, arm_access_token, cloud='None'):
        """Class ArmTokenAuthentification constructor.

        :param arm_access_token: An ARM access token.
        :type arm_access_token: str
        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
            If no default is found, "AzureCloud" is used.
        :type cloud: str
        """
        super(ArmTokenAuthentication, self).__init__(cloud)
        self._arm_access_token = arm_access_token

    def update_arm_access_token(self, new_arm_access_token):
        """Update ARM access token.

        :param new_arm_access_token: An ARM access token.
        :type new_arm_access_token: str
        """
        self._arm_access_token = new_arm_access_token

    def _get_arm_token(self):
        """Return arm access token.

        :return: Returns arm access token
        :rtype: str
        """
        return self._arm_access_token

    def _get_graph_token(self):
        raise AuthenticationException("_get_graph_token not yet supported.")

    def _get_azureml_client_token(self):
        raise AuthenticationException("_get_azureml_client_token not yet supported.")

    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        :return: Returns a list of SubscriptionInfo named tuples.
        :rtype: list, str
        """
        arm_token = self._get_arm_token()
        from azureml._vendor.azure_cli_core.auth.old_adal_authentication import AdalAuthentication
        from azureml._vendor.azure_cli_core.auth.util import resource_to_scopes
        scopes = resource_to_scopes(self._cloud_type.endpoints.active_directory_resource_id)
        token_expiry = {"expires_on": int(_get_exp_time(arm_token))}
        auth_object = AdalAuthentication(lambda x: ("Bearer", arm_token, token_expiry))
        from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token
        token_tenant_id = fetch_tenantid_from_aad_token(arm_token)

        return _get_subscription_ids_via_client(
            auth_object, scopes=scopes, resource_manager=self._cloud_type.endpoints.resource_manager
        ), token_tenant_id

    @staticmethod
    def _initialize_arm_token_auth(tenant_id, cloud=None):
        try:
            from azureml.core import Run
            secret_name = os.environ["AZUREML_AAD_TOKEN_SECRET_NAME"]
            arm_token = Run.get_context().get_secret(secret_name)
            return ArmTokenAuthentication(arm_token, cloud=cloud)
        except Exception as e:
            module_logger.debug(e)


class _DatabricksClusterAuthentication(ArmTokenAuthentication):

    def _get_arm_token(self):
        """Return arm token.

        :return: Returns the arm token.
        :rtype: str
        """
        # retrieves the AAD token through DB cluster exposed token
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().adlsAadToken().get() # noqa

    @staticmethod
    def _initialize_adb_auth(tenant_id_to_check=None):
        """Return _DatabricksClusterAuthentication object or none.

        :param tenant_id_to_check: tenant id to check, if tenant id in token matches with this tenant id.
        :type tenant_id_to_check: str
        :return: MsiAuthentication object.
        :rtype: azureml.core.authentication._DatabricksClusterAuthentication
        """
        try:
            db_auth = _DatabricksClusterAuthentication(None)
            # make sure we can get an arm token through msi to validate the auth object.
            arm_token = db_auth._get_arm_token()
            # If a user has specified a tenant id then we need to check if this token is for that tenant.
            auth_tenant = fetch_tenantid_from_aad_token(arm_token)
            if tenant_id_to_check and auth_tenant != tenant_id_to_check:
                raise UserErrorException("The tenant in the Databricks cluster authentication {} is different "
                                         "than the user specified tenant {}.".format(auth_tenant, tenant_id_to_check))
            else:
                return db_auth
        except Exception as e:
            # let it fail silently and move on
            module_logger.debug(e)


def _sp_auth_caching_decorator(token_type):

    def actual_decorator(actual_function):
        """Actual decorator.

        :param actual_function: Actual function to which this decorator was applied to.
        :type actual_function: object
        :return: Returns the wrapper.
        :rtype: object
        """
        def wrapper(self, *args, **kwargs):
            """Wrapper.

            :param args:
            :type args: list
            :param kwargs:
            :type kwargs: dict
            :return: Returns the test function.
            :rtype: object
            """
            field_name = ServicePrincipalAuthentication._token_type_to_field_dict[token_type]
            if self._enable_caching:
                cached_token = getattr(self, field_name)
                if not cached_token or self._is_token_expired(cached_token):
                    with ServicePrincipalAuthentication._sp_auth_lock:
                        # Getting it again after acquiring the lock in case some other thread might have updated it.
                        cached_token = getattr(self, field_name)
                        if not cached_token or self._is_token_expired(cached_token):
                            s = time.time()
                            module_logger.debug("Calling {} in ServicePrincipalAuthentication "
                                                "to get token.".format(actual_function))
                            new_token = actual_function(self, *args, **kwargs)
                            module_logger.debug("{} call completed in {} s".format(
                                actual_function, (time.time()-s)))
                            setattr(self, field_name, new_token)
                            return new_token
                        else:
                            return cached_token
                else:
                    return cached_token
            else:
                return actual_function(self, *args, **kwargs)

        return wrapper

    return actual_decorator


class ServicePrincipalAuthentication(AbstractAuthentication):
    """Manages authentication using a service principle instead of a user identity.

    Service Principal authentication is suitable for automated workflows like for CI/CD scenarios.
    This type of authentication decouples the authentication process from any specific user login, and
    allows for managed access control.

    .. remarks::

        Service principal authentication involves creating an App Registration in Azure Active Directory. First,
        you generate a client secret, and then you grant your service principal role access to your machine
        learning workspace. Then, you use the ServicePrincipalAuthentication class to manage your authentication
        flow.

        .. code-block:: python

            import os
            from azureml.core.authentication import ServicePrincipalAuthentication

            svc_pr_password = os.environ.get("AZUREML_PASSWORD")

            svc_pr = ServicePrincipalAuthentication(
                tenant_id="my-tenant-id",
                service_principal_id="my-application-id",
                service_principal_password=svc_pr_password)


            ws = Workspace(
                subscription_id="my-subscription-id",
                resource_group="my-ml-rg",
                workspace_name="my-ml-workspace",
                auth=svc_pr
                )

            print("Found workspace {} at location {}".format(ws.name, ws.location))

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb


        To learn about creating a service principal and allowing the service principal to access a machine
        learning workspace, see `Set up service principal authentication <https://docs.microsoft.com/en-us/azure/
        machine-learning/how-to-setup-authentication#set-up-service-principal-authentication>`_.

    :param tenant_id: The active directory tenant that the service identity belongs to.
    :type tenant_id: str
    :param service_principal_id: The service principal ID.
    :type service_principal_id: str
    :param service_principal_password: The service principal password/key.
    :type service_principal_password: str
    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, "AzureCloud" is used.
    :type cloud: str
    """

    _token_type_to_field_dict = {
        "ARM_TOKEN": "_cached_arm_token",
        "GRAPH_TOKEN": "_cached_graph_token",
        "AZUREML_CLIENT_TOKEN": "_cached_azureml_client_token"}
    _sp_auth_lock = threading.Lock()

    def __init__(self, tenant_id, service_principal_id, service_principal_password, cloud='AzureCloud', _enable_caching=True):
        """Class ServicePrincipalAuthentication constructor.

        :param tenant_id: The active directory tenant that the service identity belongs to.
        :type tenant_id: str
        :param service_principal_id: The service principal id.
        :type service_principal_id: str
        :param service_principal_password: The service principal password/key.
        :type service_principal_password: str
        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, "AzureCloud" is used.
        :type cloud: str
        """
        self._tenant_id = tenant_id
        self._service_principal_id = service_principal_id
        self._service_principal_password = service_principal_password
        self._enable_caching = _enable_caching
        self._cached_arm_token = None
        self._cached_graph_token = None
        self._cached_azureml_client_token = None
        super(ServicePrincipalAuthentication, self).__init__(cloud)

    @_sp_auth_caching_decorator("ARM_TOKEN")
    def _get_arm_token(self):
        """Return arm access token.

        :return: Returns arm token from sp credential object
        :rtype: str
        """
        resource = self._cloud_type.endpoints.active_directory_resource_id
        scopes = resource_to_scopes(resource)
        token = execute_func(self._get_sp_credential_object().get_token, *scopes)
        return token.token

    @_sp_auth_caching_decorator("GRAPH_TOKEN")
    def _get_graph_token(self):
        """Return Graph access token.

        :return: Returns Graph token from sp credential object
        :rtype: str
        """
        resource = self._cloud_type.endpoints.active_directory_graph_resource_id
        scopes = resource_to_scopes(resource)
        token = execute_func(self._get_sp_credential_object().get_token, *scopes)
        return token.token

    @_sp_auth_caching_decorator("AZUREML_CLIENT_TOKEN")
    def _get_azureml_client_token(self):
        """Return Azureml client token using azure API calls.

        :return: Azureml client token.
        :rtype: str
        """
        resource = self._get_aml_resource_id()
        scopes = resource_to_scopes(resource)
        token = execute_func(self._get_sp_credential_object().get_token, *scopes)
        return token.token

    def _get_sp_credential_object(self):
        """Return service principal credentials.

        :return: Returns sp credentials
        :rtype: ServicePrincipalCredential
        """
        from azureml._vendor.azure_cli_core.auth.msal_authentication import ServicePrincipalCredential
        from azureml._vendor.azure_cli_core.auth.identity import ServicePrincipalAuth

        # TODO: We have some sad deepcopy in our code, and keeping
        # sp_credentials object as a class field prevents the deepcopy
        # off  auth object in workspace, project classes.
        # Because sp_credentials somehow contains RLock object that cannot be deepcopied.
        login_endpoint = self._cloud_type.endpoints.active_directory
        entry = ServicePrincipalAuth.build_credential(secret_or_certificate=self._service_principal_password)
        sp_auth = ServicePrincipalAuth.build_from_credential(
            tenant_id=self._tenant_id, client_id=self._service_principal_id,
            credential=entry
        )
        sp_credentials = ServicePrincipalCredential(
            sp_auth, authority=urljoin(login_endpoint, self._tenant_id)
        )
        return sp_credentials

    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        :return: Returns a list of SubscriptionInfo named tuples.
        :rtype: list, str
        """
        arm_token = self._get_arm_token()
        arm_auth = ArmTokenAuthentication(arm_token, self._cloud_type.name)
        return arm_auth._get_all_subscription_ids()

    def _is_token_expired(self, token_to_check):
        if not token_to_check:
            return True
        try:
            decoded_json = jwt.decode(token_to_check, options={'verify_signature': False, 'verify_aud': False})
        # catch exception when token has already been expired 
        except jwt.exceptions.ExpiredSignatureError:
            return True
        else:
            return (decoded_json["exp"] - time.time()) < (5 * 60)

    @staticmethod
    def _initialize_sp_auth(tenant_id, cloud=None):
        try:
            if tenant_id is None:
                tenant_id = os.environ[_AZUREML_SERVICE_PRINCIPAL_TENANT_ID_ENV_VAR]

            service_principal_id = os.environ[_AZUREML_SERVICE_PRINCIPAL_ID_ENV_VAR]
            service_principal_password = os.environ[_AZUREML_SERVICE_PRINCIPAL_PASSWORD_ENV_VAR]

            return ServicePrincipalAuthentication(
                tenant_id, service_principal_id, service_principal_password, cloud=cloud
            )
        except Exception as e:
            module_logger.debug(e)


class AzureMLTokenAuthentication(AbstractAuthentication):
    """Manages authentication and access tokens in the context of submitted runs.

    The Azure Machine Learning token is generated when a run is submitted and is only available to the
    code that submitted the run. The AzureMLTokenAuthentication class can only be used in the context of the
    submitted run. The returned token cannot be used against any Azure Resource Manager (ARM) operations
    like provisioning compute. The Azure Machine Learning token is useful when executing a program
    remotely where it might be unsafe to use the private credentials of a user.

    .. remarks::

        Consumers of this class should call the class method :meth:`create`, which creates a new object or
        returns a registered instance with the same run_scope (``subscription_id``, ``resource_group_name``,
        ``workspace_name``, ``experiment_name``, ``run_id``) provided.

    :param azureml_access_token: The Azure ML token is generated when a run is submitted and is only
        available to the code submitted.
    :type azureml_access_token: str
    :param expiry_time: The Azure ML token's expiration time.
    :type expiry_time: datetime.datetime
    :param host:
    :type host: str
    :param subscription_id: The Azure subscription ID where the experiment is submitted.
    :type subscription_id: str
    :param resource_group_name: The resource group name where the experiment is submitted.
    :type resource_group_name: str
    :param workspace_name: The workspace where the experiment is submitted.
    :type workspace_name: str
    :param experiment_name: The experiment name.
    :type experiment_name: str
    :param experiment_id: The experiment id. If provided experiment_name will be ignored
    :type experiment_id: str
    :param run_id: The ID of the run.
    :type run_id: str
    :param user_email: Optional user email.
    :type user_email: str
    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, "AzureCloud" is used.
    :type cloud: str

    Attributes:
        EXPIRATION_THRESHOLD_IN_SECONDS (int): Seconds before expiration that refresh process starts.

        REFRESH_INTERVAL_IN_SECONDS (int): Seconds before a retry times out.

    """

    _registered_auth = dict()
    _host_clientbase = dict()
    _register_auth_lock = threading.Lock()
    _daemon = None

    # To refresh the token as late as (3 network retries (30s call timeout) + 5s buffer) seconds before it expires
    EXPIRATION_THRESHOLD_IN_SECONDS = 95
    REFRESH_INTERVAL_IN_SECONDS = 30

    def __init__(self, azureml_access_token, expiry_time=None, host=None, subscription_id=None,
                 resource_group_name=None, workspace_name=None, experiment_name=None, run_id=None, user_email=None,
                 experiment_id=None, cloud='AzureCloud'):
        """Authorize users by their Azure ML token.

        The Azure ML token is generated when a run is submitted and is only available to the code submitted.
        The class can only be used in the context of the submitted run. The token cannot be used against any ARM
        operations like provisioning compute. The Azure ML token is useful when executing a program remotely
        where it might be unsafe to use the user's private credentials. The consumer of this class should call the
        class method create which creates a new object or returns a registered instance with the same run_scope
        (subscription_id, resource_group_name, workspace_name, experiment_name, run_id) provided.

        :param azureml_access_token: The Azure ML token is generated when a run is submitted and is only
            available to the code submitted.
        :type azureml_access_token: str
        :param expiry_time: The Azure ML token's expiration time.
        :type expiry_time: datetime.Datetime
        :param host:
        :type host: str
        :param subscription_id: The Azure subscription ID where the experiment is submitted.
        :type subscription_id: str
        :param resource_group_name: The resource group name where the experiment is submitted.
        :type resource_group_name: str
        :param workspace_name: The workspace where the experiment is submitted.
        :type workspace_name: str
        :param experiment_name: The experiment name.
        :type experiment_name: str
        :param experiment_id: The experiment id. If provided experiment_name will be ignored
        :type experiment_id: str
        :param run_id: The ID of the run.
        :type run_id: str
        :param user_email: An optional user email.
        :type user_email: str
        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, "AzureCloud" is used.
        :type cloud: str
        """
        self._aml_access_token = azureml_access_token
        self._user_email = user_email
        self._aml_token_lock = LoggedLock(_ident="AMLTokenLock", _parent_logger=module_logger)
        self._expiry_time = AzureMLTokenAuthentication._parse_expiry_time_from_token(
            self._aml_access_token) if expiry_time is None else expiry_time
        self._host = host
        self._subscription_id = subscription_id
        self._resource_group_name = resource_group_name
        self._workspace_name = workspace_name
        self._experiment_name = experiment_name
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._run_scope_info = (self._subscription_id,
                                self._resource_group_name,
                                self._workspace_name,
                                self._experiment_id if self._experiment_id else self._experiment_name,
                                self._run_id)
        super(AzureMLTokenAuthentication, self).__init__(cloud)

        if AzureMLTokenAuthentication._daemon is None:
            AzureMLTokenAuthentication._daemon = Daemon(work_func=AzureMLTokenAuthentication._update_registered_auth,
                                                        interval_sec=AzureMLTokenAuthentication.REFRESH_INTERVAL_IN_SECONDS,
                                                        _ident="TokenRefresherDaemon",
                                                        _parent_logger=module_logger)
            AzureMLTokenAuthentication._daemon.start()

        if any((param is None for param in self._run_scope_info)):
            module_logger.warning("The AzureMLTokenAuthentication created will not be updated due to missing params. "
                                  "The token expires on {}.\n".format(self._expiry_time))
        else:
            self._register_auth()

    @classmethod
    def create(cls, azureml_access_token, expiry_time, host, subscription_id,
               resource_group_name, workspace_name, experiment_name, run_id, user_email=None, experiment_id=None):
        """Create an AzureMLTokenAuthentication object or return a registered instance with the same run_scope.

        :param cls: Indicates class method.
        :param azureml_access_token: The Azure ML token is generated when a run is submitted and is only
            available to the code submitted.
        :type azureml_access_token: str
        :param expiry_time: The Azure ML token's expiration time.
        :type expiry_time: datetime.datetime
        :param host:
        :type host: str
        :param subscription_id: The Azure subscription ID where the experiment is submitted.
        :type subscription_id: str
        :param resource_group_name: The resource group name where the experiment is submitted.
        :type resource_group_name: str
        :param workspace_name: The workspace where the experiment is submitted.
        :type workspace_name: str
        :param experiment_name: The experiment name.
        :type experiment_name: str
        :param experiment_id: The experiment id. If provided experiment_name will be ignored
        :type experiment_id: str
        :param run_id: The ID of the run.
        :type run_id: str
        :param user_email: An optional user email.
        :type user_email: str
        """
        auth_key = cls._construct_key(subscription_id,
                                      resource_group_name,
                                      workspace_name,
                                      experiment_id if experiment_id else experiment_name,
                                      run_id)
        if auth_key in cls._registered_auth:
            return cls._registered_auth[auth_key]

        return cls(azureml_access_token, expiry_time, host, subscription_id,
                   resource_group_name, workspace_name, experiment_name, run_id, user_email, experiment_id)

    @staticmethod
    def _parse_expiry_time_from_token(token):
        # We set verify=False, as we don't have keys to verify signature, and we also don't need to
        # verify signature, we just need the expiry time.
        decode_json = jwt.decode(token, options={'verify_signature': False, 'verify_aud': False})
        return AzureMLTokenAuthentication._convert_to_datetime(decode_json['exp'])

    @staticmethod
    def _convert_to_datetime(expiry_time):
        if isinstance(expiry_time, datetime.datetime):
            return expiry_time
        try:
            date = dateutil.parser.parse(expiry_time)
        except (ValueError, TypeError):
            date = datetime.datetime.fromtimestamp(int(expiry_time))
        return date

    @staticmethod
    def _get_token_dir():
        temp_dir = os.environ.get("AZ_BATCHAI_JOB_TEMP", None)
        if not temp_dir:
            return None
        else:
            return os.path.join(temp_dir, "run_token")

    @staticmethod
    def _token_encryption_enabled():
        return not os.environ.get("AZUREML_DISABLE_REFRESHED_TOKEN_SHARING") and \
               os.environ.get("AZUREML_RUN_TOKEN_PASS") is not None and \
               os.environ.get("AZUREML_RUN_TOKEN_RAND") is not None

    @staticmethod
    def _get_token(token, should_encrypt=False):
        password = os.environ.get("AZUREML_RUN_TOKEN_PASS")
        random_string = os.environ.get("AZUREML_RUN_TOKEN_RAND")
        m = hashlib.sha256()
        m.update(random_string.encode())
        salt = m.digest()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        f = Fernet(key)
        if should_encrypt:
            return f.encrypt(token.encode()).decode()
        else:
            return f.decrypt(token.encode()).decode()

    @staticmethod
    def _encrypt_token(token):
        return AzureMLTokenAuthentication._get_token(token, should_encrypt=True)

    @staticmethod
    def _decrypt_token(token):
        return AzureMLTokenAuthentication._get_token(token, should_encrypt=False)

    @staticmethod
    def _get_initial_token_and_expiry():
        token = os.environ['AZUREML_RUN_TOKEN']
        token_expiry_time = os.environ.get('AZUREML_RUN_TOKEN_EXPIRY',
                                           AzureMLTokenAuthentication._parse_expiry_time_from_token(token))
        # The token dir and the token file are only created when the token expires and the token refresh happens.
        # If the token dir and the token file don't exist, that means that the token has not expired yet and
        # we should continue to use the token value from the env var.
        # make reading/writing a token file best effort initially
        if AzureMLTokenAuthentication._token_encryption_enabled():
            try:
                latest_token_file = None
                token_dir = AzureMLTokenAuthentication._get_token_dir()
                if token_dir and os.path.exists(token_dir):
                    token_files = [f for f in os.listdir(token_dir) if
                                   os.path.isfile(os.path.join(token_dir, f)) and "token.txt" in f]
                    if len(token_files) > 0:
                        convert = lambda text: int(text) if text.isdigit() else text
                        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
                        token_files.sort(key=alphanum_key, reverse=True)
                        latest_token_file = token_files[0]

                if latest_token_file is not None:
                    latest_token_file_full_path = os.path.join(token_dir, latest_token_file)
                    if os.path.exists(latest_token_file_full_path):
                        module_logger.debug("Reading token from:{}".format(latest_token_file_full_path))
                        encrypted_token = token
                        with open(latest_token_file_full_path, "r") as file_handle:
                            encrypted_token = file_handle.read()
                        token = AzureMLTokenAuthentication._decrypt_token(encrypted_token)
                        token_expiry_time = AzureMLTokenAuthentication._parse_expiry_time_from_token(token)
            except Exception as ex:
                module_logger.debug("Exception while reading a token:{}".format(ex))
        return token, token_expiry_time

    @property
    def token(self):
        """Return the Azure ML token.

        :return: The Azure ML access token.
        :rtype: str
        """
        with self._aml_token_lock:
            return self._aml_access_token

    @property
    def expiry_time(self):
        """Return the Azure ML token's expiration time.

        :return: The expiration time.
        :rtype: datetime.datetime
        """
        with self._aml_token_lock:
            return self._expiry_time

    def get_authentication_header(self):
        """Return the HTTP authorization header.

        The authorization header contains the user access token for access authorization against the service.

        :return: Returns the HTTP authorization header.
        :rtype: dict
        """
        return {"Authorization": "Bearer " + self._aml_access_token}

    def set_token(self, token, expiry_time):
        """Update Azure ML access token.

        :param token: The token to refresh.
        :type token: str
        :param expiry_time: The new expiration time.
        :type expiry_time: datetime.datetime
        """
        self._aml_access_token = token
        self._expiry_time = expiry_time
        # make reading/writing a token file best effort initially
        if AzureMLTokenAuthentication._token_encryption_enabled():
            try:
                token_dir = AzureMLTokenAuthentication._get_token_dir()
                if token_dir:
                    module_logger.debug("Token directory {}".format(token_dir))
                    encrypted_token = AzureMLTokenAuthentication._encrypt_token(token)
                    seconds = (datetime.datetime.utcnow() - datetime.datetime(1, 1, 1)).total_seconds()
                    if not os.path.exists(token_dir):
                        try:
                            os.makedirs(token_dir, exist_ok=True)
                        except OSError as ex:
                            if ex.errno != errno.EEXIST:
                                raise
                    token_file_path = os.path.join(token_dir, "{}_{}_token.txt".format(seconds, os.getpid()))
                    module_logger.debug("Token file {}".format(token_file_path))
                    with open(token_file_path, "w") as file:
                        file.write(encrypted_token)
            except Exception as ex:
                module_logger.debug("Exception while writing a token:{}".format(ex))

    def _get_arm_token(self):
        raise AuthenticationException("AzureMLTokenAuthentication._get_arm_token "
                                      "not yet supported.")

    def _get_graph_token(self):
        raise AuthenticationException("AzureMLTokenAuthentication._get_graph_token "
                                      "not yet supported.")

    def _get_azureml_client_token(self):
        raise AuthenticationException("AzureMLTokenAuthentication._get_azureml_client_token "
                                      "not yet supported.")

    def _get_workspace(self, subscription_id, resource_group, name, **kwargs):
        from azureml.core import Workspace

        if subscription_id is not None and subscription_id != self._subscription_id \
                or resource_group is not None and resource_group != self._resource_group_name \
                or name is not None and name != self._workspace_name:
            return None

        override_host = kwargs.get("override_host", None)
        workspace_id = kwargs.get("workspace_id", None)

        try:
            url = override_host if override_host else os.environ["AZUREML_SERVICE_ENDPOINT"]
            workspace_id = workspace_id if workspace_id else os.environ["AZUREML_WORKSPACE_ID"]

            location_from_url_regex_match = re.compile(r"//(.*?)\.").search(url)
            location = location_from_url_regex_match.group(1) if location_from_url_regex_match else None
            workspace_discovery_url = os.environ.get(DISCOVERY_END_POINT, None)

            if workspace_discovery_url is None:
                module_logger.debug("Discovery url not found will try to get workspace to fetch discovery url")
                from azureml._project._commands import _get_workspace_dp_from_base_url
                autorest_ws = _get_workspace_dp_from_base_url(
                    auth=self, subscription_id=subscription_id, resource_group=resource_group,
                    workspace_name=name, base_url=url
                )
                workspace_discovery_url = autorest_ws.discovery_url
                location = autorest_ws.location
                module_logger.debug(
                    "Discovery url set to {} and location set to {}".format(workspace_discovery_url, location))

        except Exception as e:
            raise_from(RunEnvironmentException(), e)

        # Disabling service check, as this is in remote context and we don't have an ARM token
        # to check ARM if the workspace exists or not.
        service_context = ServiceContext(subscription_id,
                                         resource_group,
                                         name,
                                         workspace_id,
                                         workspace_discovery_url,
                                         self)
        return Workspace._from_service_context(service_context, location)

    @staticmethod
    def _construct_key(*args):
        return "//".join(args)

    @staticmethod
    def _should_refresh_token(current_expiry_time):
        if current_expiry_time is None:
            return True
        # Refresh when remaining duration < EXPIRATION_THRESHOLD_IN_SECONDS
        expiry_time_utc = current_expiry_time.replace(tzinfo=pytz.utc)
        current_time = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        time_difference = expiry_time_utc - current_time
        time_to_expire = time_difference / datetime.timedelta(seconds=1)
        module_logger.debug("Time to expire {} seconds".format(time_to_expire))
        return time_to_expire < AzureMLTokenAuthentication.EXPIRATION_THRESHOLD_IN_SECONDS

    def _update_auth(self):
        if AzureMLTokenAuthentication._should_refresh_token(self.expiry_time):
            module_logger.debug("Expiration time for run scope: {} = {}, refreshing token\n".format(
                self._run_scope_info, self.expiry_time))
            with AzureMLTokenAuthentication._register_auth_lock:
                host_key = AzureMLTokenAuthentication._construct_key(self._host,
                                                                     self._subscription_id,
                                                                     self._resource_group_name,
                                                                     self._workspace_name)
                if host_key in AzureMLTokenAuthentication._host_clientbase:
                    if self._experiment_id:
                        token_result = AzureMLTokenAuthentication._host_clientbase[host_key]._client.run.get_token_by_exp_id(
                            *self._run_scope_info,
                            is_async=False
                        )
                    else:
                        token_result = AzureMLTokenAuthentication._host_clientbase[host_key]._client.run.get_token(
                            *self._run_scope_info,
                            is_async=False
                        )
                    self.set_token(token_result.token, token_result.expiry_time_utc)

    def _register_auth(self):
        auth_key = AzureMLTokenAuthentication._construct_key(
            self._subscription_id,
            self._resource_group_name,
            self._workspace_name,
            self._experiment_id if self._experiment_id else self._experiment_name,
            self._run_id
        )
        if auth_key in AzureMLTokenAuthentication._registered_auth:
            module_logger.warning("Already registered authentication for run id: {}".format(self._run_id))
        else:
            from azureml._restclient.token_client import TokenClient
            host_key = AzureMLTokenAuthentication._construct_key(self._host,
                                                                 self._subscription_id,
                                                                 self._resource_group_name,
                                                                 self._workspace_name)
            if host_key not in AzureMLTokenAuthentication._host_clientbase:
                AzureMLTokenAuthentication._host_clientbase[host_key] = TokenClient(self, self._host)

            self._update_auth()
            with AzureMLTokenAuthentication._register_auth_lock:
                AzureMLTokenAuthentication._registered_auth[auth_key] = self

    @classmethod
    def _update_registered_auth(cls):
        with cls._register_auth_lock:
            auths = list(cls._registered_auth.values())
        for auth in auths:
            auth._update_auth()

    @staticmethod
    def _initialize_aml_token_auth(tenant_id):
        try:
            # Only use the run environment for ambient auth if specifically requested,
            # because the AML token doesn't work for ARM operations.
            if os.environ.get("AZUREML_RUN_TOKEN_USE_AMBIENT_AUTH", "").lower() != "true":
                return None

            # Load authentication scope environment variables.
            subscription_id = os.environ['AZUREML_ARM_SUBSCRIPTION']
            resource_group = os.environ["AZUREML_ARM_RESOURCEGROUP"]
            workspace_name = os.environ["AZUREML_ARM_WORKSPACE_NAME"]
            experiment_name = os.environ["AZUREML_ARM_PROJECT_NAME"]
            experiment_id = os.environ.get("AZUREML_EXPERIMENT_ID", None)
            run_id = os.environ["AZUREML_RUN_ID"]
            url = os.environ["AZUREML_SERVICE_ENDPOINT"]

            # Initialize an AMLToken auth, authorized for the current run.
            token, token_expiry_time = AzureMLTokenAuthentication._get_initial_token_and_expiry()

            return AzureMLTokenAuthentication.create(token,
                                                     AzureMLTokenAuthentication._convert_to_datetime(
                                                         token_expiry_time),
                                                     url,
                                                     subscription_id,
                                                     resource_group,
                                                     workspace_name,
                                                     experiment_name,
                                                     run_id,
                                                     experiment_id=experiment_id)
        except Exception as e:
            module_logger.debug(e)
            return None


class MsiAuthentication(AbstractAuthentication):
    """Manages authentication using a managed identity in Azure Active Directory.

    When using Azure ML SDK on Azure Virtual Machine (VM), you can authenticate with a `managed
    identity <https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/overview>`_
    (formerly known as Managed Service Identity - MSI). Using a managed identity allows the VM connect
    to your workspace without storing credentials in Python code, thus decoupling the authentication
    process from any specific user login.

    .. remarks::

        The following example shows how to use MsiAuthentication.

        .. code-block:: python

            from azureml.core.authentication import MsiAuthentication

            msi_auth = MsiAuthentication()

            ws = Workspace(subscription_id="my-subscription-id",
                           resource_group="my-ml-rg",
                           workspace_name="my-ml-workspace",
                           auth=msi_auth)

            print("Found workspace {} at location {}".format(ws.name, ws.location))

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb


    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
        If no default is found, "AzureCloud" is used.
    :type cloud: str
    :param identity_config: a mapping ``{parameter_name: value}`` specifying a user-assigned identity by its object
      or resource ID, for example ``{"client_id": "..."}``. Check the documentation for your hosting environment to
      learn what values it expects.
    :type identity_config: Mapping[str, str]
    """

    def __init__(self, cloud=None, **kwargs):
        """Class MsiAuthentication constructor.

        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
            If no default is found, "AzureCloud" is used.
        :type cloud: str
        :param identity_config: a mapping ``{parameter_name: value}`` specifying a user-assigned identity by its object
            or resource ID, for example ``{"client_id": "..."}``. Check the documentation for your hosting environment to
            learn what values it expects.
        :type identity_config: Mapping[str, str]
        """
        super(MsiAuthentication, self).__init__(cloud)
        self._identity_config = kwargs.pop("identity_config", None) or {}

    def _get_resource_token(self, resource):
        """Return resource token.

        :return: Returns the resource token.
        :rtype: str
        """
        # retrieves the AAD token through MSI
        from azureml._vendor.azure_cli_core.auth.adal_authentication import MSIAuthenticationWrapper
        msi_auth = MSIAuthenticationWrapper(
            resource=resource, cloud_environment=self._cloud_type, **self._identity_config
        )
        msi_auth.set_token()
        return msi_auth.token['access_token']

    def _get_arm_token(self):
        """Return an ARM token.

        :return: Returns the ARM token.
        :rtype: str
        """
        # retrieves the AAD token through MSI
        from azureml._vendor.azure_cli_core.auth.adal_authentication import MSIAuthenticationWrapper
        msi_auth = MSIAuthenticationWrapper(cloud_environment=self._cloud_type, **self._identity_config)
        msi_auth.set_token()
        return msi_auth.token['access_token']

    def _get_graph_token(self):
        """Return graph token.

        :return: Returns the graph token.
        :rtype: str
        """
        # retrieves the AAD token through MSI
        from azureml._vendor.azure_cli_core.auth.adal_authentication import MSIAuthenticationWrapper
        resource = self._cloud_type.endpoints.active_directory_graph_resource_id
        msi_auth = MSIAuthenticationWrapper(resource, cloud_environment=self._cloud_type, **self._identity_config)
        msi_auth.set_token()
        return msi_auth.token['access_token']

    def _get_azureml_client_token(self):
        """Return azureml client token.

        :return: Returns the graph token.
        :rtype: str
        """
        # retrieves the AAD token through MSI
        from azureml._vendor.azure_cli_core.auth.adal_authentication import MSIAuthenticationWrapper
        resource = self._get_aml_resource_id()
        msi_auth = MSIAuthenticationWrapper(
            resource=resource, cloud_environment=self._cloud_type, **self._identity_config
        )
        msi_auth.set_token()
        return msi_auth.token['access_token']

    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        :return: Returns a list of SubscriptionInfo named tuples.
        :rtype: list
        """
        from azureml._vendor.azure_cli_core.auth.adal_authentication import MSIAuthenticationWrapper
        from azureml._vendor.azure_cli_core.auth.util import resource_to_scopes
        scopes = resource_to_scopes(self._cloud_type.endpoints.active_directory_resource_id)
        msi_auth = MSIAuthenticationWrapper(cloud_environment=self._cloud_type, **self._identity_config)
        arm_token = self._get_arm_token()
        from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token
        token_tenant_id = fetch_tenantid_from_aad_token(arm_token)
        return _get_subscription_ids_via_client(
            msi_auth, scopes=scopes,
            resource_manager=self._cloud_type.endpoints.resource_manager
        ), token_tenant_id

    @staticmethod
    def _initialize_msi_auth(tenant_id_to_check=None, cloud=None):
        """Return MsiAuthentication object or none.

        :param tenant_id_to_check: tenant id to check, if tenant id in token matches with this tenant id.
        :type tenant_id_to_check: str
        :return: MsiAuthentication object.
        :rtype: azureml.core.authentication.MsiAuthentication
        """
        try:
            nbvm_file_path = "/mnt/azmnt/.nbvm"
            if "MSI_ENDPOINT" in os.environ or os.path.isfile(nbvm_file_path):
                # MSI_ENDPOINT is always set by Azure Notebooks, so if the MSI_ENDPOINT env var
                # is set, we will try to create an MsiAuthentication object.
                # Similarly, on DSI/Notebook VM, the path /mnt/azmnt/.nbvm will exist,
                # and we should try to use MSI authentication.
                msi_auth = MsiAuthentication(cloud=cloud)
                # make sure we can get an arm token through msi to validate the auth object.
                arm_token = msi_auth._get_arm_token()
                # If a user has specified a tenant id then we need to check if this token is for that tenant.
                if tenant_id_to_check and fetch_tenantid_from_aad_token(arm_token) != tenant_id_to_check:
                    raise UserErrorException("The tenant in the MSI authentication is different "
                                             "than the user specified tenant.")
                else:
                    return msi_auth
        except Exception as e:
            # let it fail silently and move on to AzureCliAuthentication as users on
            # Azure Notebooks/NBVM may want to use the credentials from 'az login'.
            module_logger.debug(e)


class _ArcadiaAuthentication(ArmTokenAuthentication):
    """Authentication class for Arcadia cluster."""

    _cached_arm_token = None

    _cached_graph_token = None

    _ARCADIA_ENVIRONMENT_VARIABLE_NAME = "AZURE_SERVICE"
    _ARCADIA_ENVIRONMENT_VARIABLE_VALUE = "Microsoft.ProjectArcadia"

    def __init__(self, cloud=None):
        super(_ArcadiaAuthentication, self).__init__(None, cloud)

    def _get_arm_token(self):
        """Return arm token.

        :return: Returns the arm token.
        :rtype: str
        """
        from azureml._base_sdk_common._arcadia_token_wrapper import PyTokenLibrary
        if _ArcadiaAuthentication._cached_arm_token and \
                not ((_get_exp_time(_ArcadiaAuthentication._cached_arm_token) - time.time())
                     < _TOKEN_REFRESH_THRESHOLD_SEC):
            return _ArcadiaAuthentication._cached_arm_token
        else:
            _ArcadiaAuthentication._cached_arm_token = PyTokenLibrary.get_AAD_token(PyTokenLibrary._ARM_RESOURCE)
            return _ArcadiaAuthentication._cached_arm_token

    @staticmethod
    def _initialize_arcadia_auth(tenant_id_to_check=None, cloud=None):
        """Return _ArcadiaAuthentication object or none.

        :param tenant_id_to_check: tenant id to check, if tenant id in token matches with this tenant id.
        :type tenant_id_to_check: str
        :return: _ArcadiaAuthentication object.
        :rtype: azureml.core.authentication._ArcadiaAuthentication
        """
        try:
            if _ArcadiaAuthentication._is_arcadia_environment():
                arcadia_auth = _ArcadiaAuthentication(cloud=cloud)
                arm_token = arcadia_auth._get_arm_token()
                # If a user has specified a tenant id then we need to check if this token is for that tenant.
                if tenant_id_to_check and fetch_tenantid_from_aad_token(arm_token) != tenant_id_to_check:
                    raise UserErrorException("The tenant in the Arcadia cluster is different "
                                             "than the user specified tenant.")
                else:
                    return arcadia_auth
        except Exception as e:
            module_logger.debug(e)

    @staticmethod
    def _is_arcadia_environment():
        return os.environ.get(_ArcadiaAuthentication._ARCADIA_ENVIRONMENT_VARIABLE_NAME, None) \
                    == _ArcadiaAuthentication._ARCADIA_ENVIRONMENT_VARIABLE_VALUE


class TokenAuthentication(AbstractAuthentication):
    """Manage authentication using AAD token scoped by audience.

    Token Authentication is suitable when token generation and its refresh are outside of AML SDK. This type of
    authentication allows greater control over token generation and its refresh.

    For automated workflows where managed access control is needed, use the
    :class:`azureml.core.authentication.ServicePrincipalAuthentication` instead.

    :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
        "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
        If no default is found, "AzureCloud" is used.
    :type cloud: str
    :param get_token_for_audience: Function to retrieve token.

    This class requires `get_token_for_audience` method be provided which will be called to retrieve the token.

            Example how get_token_for_audience will be called and will be passed an audience
                get_token_for_audience(audience)
    """

    def __init__(self, get_token_for_audience, cloud=None):
        """Manage authentication using AAD token scoped by audience.

        Token Authentication is suitable when token generation and its refresh are outside of AML SDK. This type of
        authentication allows greater control over token generation and its refresh.

        For automated workflows where managed access control is needed, use the
        :class:`azureml.core.authentication.ServicePrincipalAuthentication` instead.

        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified, any configured default from the Azure CLI is used.
            If no default is found, "AzureCloud" is used.
        :type cloud: str
        :param get_token_for_audience: Function to retrieve token.



        This class requires `get_token_for_audience` method be provided which will be called to retrieve the token.

                Example how get_token_for_audience will be called and will be passed an audience
                    get_token_for_audience(audience)

                where audience can be either ARM or AML
                    auth = TokenAuthentication(get_token_for_audience)

                    AML audience value passed to get_token_for_audience can be retrieved by :
                    auth.get_aml_resource_id(cloud)

                    ARM audience value passed to get_token_for_audience can be retrieved by :
                    auth._cloud_type.endpoints.active_directory_resource_id

        """
        super(TokenAuthentication, self).__init__(cloud)
        self._get_token_for_audience = get_token_for_audience
        if self._get_token_for_audience is None:
            raise AuthenticationException("get_token_for_audience cannot be null.")

    def get_token(self, audience=Audience.ARM):
        """Return the arm access token scoped by audience.

        :param audience: audience of the token to retrieve.
        :type audience: Audience
        :return: Returns the arm access token.
        :rtype: str
        """
        return self._get_aad_token(resource=audience)

    def _get_aad_token(self, resource=Audience.ARM):
        """Return the aad access token scoped by audience.

        :return: Returns the scoped aad access token.
        :rtype: str
        """
        if resource not in [Audience.ARM, Audience.AZUREML]:
            raise ValueError("Unsupported audience {0}. Supported values are : {1} and {2}".format(resource, Audience.ARM, Audience.AZUREML))

        if resource is Audience.ARM:
            resource = self._cloud_type.endpoints.active_directory_resource_id
        if resource is Audience.AZUREML:
            resource = self._get_aml_resource_id()
        return self._get_token_for_audience(resource)

    def _get_arm_token(self):
        """Return the arm access token.

        :return: Returns the arm access token.
        :rtype: str
        """
        return self._get_aad_token(resource=Audience.ARM)

    def _get_graph_token(self):
        raise AuthenticationException("_get_graph_token not yet supported.")

    def _get_all_subscription_ids(self):
        """Return a list of subscriptions that are accessible through this authentication.

        :return: Returns a list of SubscriptionInfo named tuples.
        :rtype: list, str
        """
        arm_token = self._get_arm_token()
        from azureml._vendor.azure_cli_core.auth.old_adal_authentication import AdalAuthentication
        from azureml._vendor.azure_cli_core.auth.util import resource_to_scopes
        scopes = resource_to_scopes(self._cloud_type.endpoints.active_directory_resource_id)
        token_expiry = {"expires_on": int(_get_exp_time(arm_token))}
        auth_object = AdalAuthentication(lambda x: ("Bearer", arm_token, token_expiry))
        from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token
        token_tenant_id = fetch_tenantid_from_aad_token(arm_token)

        return _get_subscription_ids_via_client(
            auth_object, scopes=scopes,
            resource_manager=self._cloud_type.endpoints.resource_manager
        ), token_tenant_id


def _get_subscription_ids_via_client(
    auth_obj, resource_manager='https://management.azure.com', scopes=['https://management.azure.com/.default'],
):
    """Return a list of subscriptions that are accessible through this authentication.

    Helper function to retrieve subscriptionIDs and names using the SubscriptionClient.

    :param auth_obj: The msrest auth object. This is not azureml.core.authentication object.
    :type auth_obj: azureml._vendor.azure_cli_core.auth.old_adal_authentication.AdalAuthentication
    :param scopes: The list of case-sensitive strings.
    :type scopes: list
    :param resource_manager: The URL to the target Resource Manager endpoint.
    :type resource_manager: str
    :return: Returns a list of SubscriptionInfo named tuples.
    :rtype: list
    """
    from azure.mgmt.resource import SubscriptionClient
    result = []
    subscription_client = SubscriptionClient(
        auth_obj, base_url=resource_manager, credential_scopes=scopes
    )
    for subscription in subscription_client.subscriptions.list():
        subscription_info = _SubscriptionInfo(subscription.display_name,
                                              subscription.subscription_id)
        result.append(subscription_info)
    return result


def _get_service_client_using_arm_token(auth, client_class, subscription_id,
                                        subscription_bound=True, base_url=None):
    """Return service client.

    :param auth:
    :type auth: AbstractAuthentication
    :param client_class: The service client class
    :type client_class: object
    :param subscription_id:
    :type subscription_id: str
    :param subscription_bound: True if a subscription id is required to construct the client. Only False for
                               1 RP clients.
    :type subscription_bound: bool
    :param base_url:
    :type base_url: str
    :return:
    :rtype: object
    """
    adal_auth_object = auth._get_adal_auth_object()

    # 1 RP clients are not subscription bound.
    if not subscription_bound:
        client = client_class(adal_auth_object, base_url=base_url)
    else:
        # converting subscription_id, which is string, to string because of weird python 2.7 errors.
        client = client_class(adal_auth_object, str(subscription_id), base_url=base_url)
    return client


@_retry_connection_aborted(retries=5)
def _get_arm_token_with_refresh(profile_object, account_object, config_object, session_object,
                                config_directory, force_reload=False, resource=None):
    """Get an ARM token while refreshing it if needed.

    This is a common function across InteractiveLoginAuthentication and AzureCliAuthentication.

    :return:
    :rtype: str
    """
    AZ_CLI_AAP_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'

    if force_reload or not account_object.data:
        account_object.load(os.path.join(config_directory, 'azureProfile.json'))
        config_object.load(os.path.join(config_directory, 'az.json'))
        session_object.load(os.path.join(config_directory, 'az.sess'),
                            max_age=AzureCliAuthentication._TOKEN_TIMEOUT_SECs)

    try:
        # profile_object.get_raw_token is being used as per the suggestion
        # https://github.com/Azure/azure-cli/issues/15805#issuecomment-722914070
        access_token = profile_object.get_raw_token(resource=resource)[0][1]
        """
        According to below documentation, MSAL takes care of refreshing old tokens. In that case, we do not
        need to explicitly test and check for token expiry, making below code redundant.
        MSAL also takes care of caching.
        https://docs.microsoft.com/en-us/azure/active-directory/develop/msal-acquire-cache-tokens#acquiring-tokens-silently-from-the-cache

        """
        return access_token
    except Exception as e:
        raise AuthenticationException("Could not retrieve user token. Please run 'az login'", inner_exception=e)


def _get_exp_time(access_token):
    """Return the expiry time of the supplied arm access token.

    :param access_token:
    :type access_token: str
    :return:
    :rtype: float
    """
    # We set verify=False, as we don't have keys to verify signature, and we also don't need to
    # verify signature, we just need the expiry time.
    decode_json = jwt.decode(access_token, options={'verify_signature': False, 'verify_aud': False})
    return decode_json['exp']


def _get_profile(cloud_type=None):
    from azureml._vendor.azure_cli_core import get_default_cli
    from azureml._vendor.azure_cli_core._profile import Profile
    from azureml._vendor.azure_cli_core.cloud import get_cloud
    cli_ctx = get_default_cli()
    if cloud_type:
        cli_ctx.cloud = get_cloud(cli_ctx, cloud_type.name)
    profile_object = Profile(cli_ctx=cli_ctx)
    return profile_object
