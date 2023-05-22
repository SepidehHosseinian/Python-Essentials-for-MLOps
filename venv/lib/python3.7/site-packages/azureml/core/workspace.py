# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing a workspace, the top-level resource in Azure Machine Learning.

This module contains the :class:`azureml.core.workspace.Workspace` class and its methods and attributes that
allows you to manage machine learning artifacts like compute targets, environments, data stores, experiments,
and models. A workspace is tied to an Azure subscription and resource group, and is the primary means for billing.
Workspaces support Azure Resource Manager role-based access control (RBAC) and region affinity for all machine
learning data saved within the workspace.
"""
from random import choice
from string import ascii_lowercase
import collections
import errno
import logging
import re
import os
from operator import attrgetter

from azureml._project import _commands
from azureml._file_utils.file_utils import normalize_path, normalize_path_and_join, \
    check_and_create_dir, traverse_up_path_and_find_file, normalize_file_ext

# Az CLI converts the keys to camelCase and our tests assume that behavior,
# so converting for the SDK too.
import azureml._project.project_info as project_info
from azureml.exceptions import WorkspaceException, ProjectSystemException, UserErrorException
from azureml._base_sdk_common.common import convert_dict_keys_to_camel_case, give_warning
from azureml._base_sdk_common.common import check_valid_resource_name
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.image import Image
from azureml.core.keyvault import Keyvault
from azureml.core.model import Model
from azureml.core.webservice import Webservice
from azureml.data.datastore_client import _DatastoreClient
from azureml.core.dataset import Dataset
from azureml.core.linked_service import LinkedService
from azureml._base_sdk_common.common import get_run_config_dir_name, get_config_file_name
from azureml._base_sdk_common.workspace.models.workspace import Workspace as _ModelsWorkspace
from azureml._base_sdk_common import __version__ as VERSION

module_logger = logging.getLogger(__name__)

_WorkspaceScopeInfo = collections.namedtuple(
    "WorkspaceScopeInfo",
    "subscription_id resource_group workspace_name")

WORKSPACE_DEFAULT_BLOB_STORE_NAME = 'workspaceblobstore'


class Workspace(object):
    """Defines an Azure Machine Learning resource for managing training and deployment artifacts.

    A Workspace is a fundamental resource for machine learning in Azure Machine Learning. You use a workspace to
    experiment, train, and deploy machine learning models. Each workspace is tied to an Azure subscription and
    resource group, and has an associated SKU.

    For more information about workspaces, see:

    * `What is a Azure Machine Learning
      workspace? <https://docs.microsoft.com/azure/machine-learning/concept-workspace>`_

    * `Manage access to a
      workspace <https://docs.microsoft.com/azure/machine-learning/how-to-assign-roles>`_

    .. remarks::

        The following sample shows how to create a workspace.

        .. code-block:: python

            from azureml.core import Workspace
            ws = Workspace.create(name='myworkspace',
                        subscription_id='<azure-subscription-id>',
                        resource_group='myresourcegroup',
                        create_resource_group=True,
                        location='eastus2'
                        )

        Set ``create_resource_group`` to False if you have an existing Azure resource group that
        you want to use for the workspace.

        To use the same workspace in multiple environments, create a JSON configuration file. The configuration
        file saves your subscription, resource, and workspace name so that it can be easily loaded. To save the
        configuration use the :meth:`write_config` method.

        .. code-block:: python

            ws.write_config(path="./file-path", file_name="ws_config.json")

        See `Create a workspace configuration
        file <https://docs.microsoft.com/azure/machine-learning/how-to-configure-environment#workspace>`_
        for an example of the configuration file.

        To load the workspace from the configuration file, use the :meth:`from_config` method.

        .. code-block:: python

            ws = Workspace.from_config()
            ws.get_details()

        Alternatively, use the :meth:`get` method to load an existing workspace without using configuration files.

        .. code-block:: python

            ws = Workspace.get(name="myworkspace",
                        subscription_id='<azure-subscription-id>',
                        resource_group='myresourcegroup')

        The samples above may prompt you for Azure authentication credentials using an interactive login dialog.
        For other use cases, including using the Azure CLI to authenticate and authentication in automated
        workflows, see `Authentication in Azure Machine Learning <https://aka.ms/aml-notebook-auth>`_.

    :param subscription_id: The Azure subscription ID containing the workspace.
    :type subscription_id: str
    :param resource_group: The resource group containing the workspace.
    :type resource_group: str
    :param workspace_name: The existing workspace name.
    :type workspace_name: str
    :param auth: The authentication object. For more details, see https://aka.ms/aml-notebook-auth.
        If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
    :type auth: azureml.core.authentication.ServicePrincipalAuthentication or
        azureml.core.authentication.InteractiveLoginAuthentication or azureml.core.authentication.MsiAuthentication
    :param _location: Internal use only.
    :type _location: str
    :param _disable_service_check: Internal use only.
    :type _disable_service_check: bool
    :param _workspace_id: Internal use only.
    :type _workspace_id: str
    :param sku: The parameter is present for backwards compatibility and is ignored.
    :type sku: str
    :param _cloud: Internal use only.
    :type _cloud: str
    """

    DEFAULT_CPU_CLUSTER_CONFIGURATION = AmlCompute.provisioning_configuration(
        min_nodes=0,
        max_nodes=2,
        vm_size="STANDARD_DS2_V2",
        vm_priority="dedicated")
    DEFAULT_CPU_CLUSTER_NAME = "cpu-cluster"

    DEFAULT_GPU_CLUSTER_CONFIGURATION = AmlCompute.provisioning_configuration(
        min_nodes=0,
        max_nodes=2,
        vm_size="STANDARD_NC6",
        vm_priority="dedicated")
    DEFAULT_GPU_CLUSTER_NAME = "gpu-cluster"

    def __init__(
            self,
            subscription_id,
            resource_group,
            workspace_name,
            auth=None,
            _location=None,
            _disable_service_check=False,
            _workspace_id=None,
            sku='basic',
            tags=None,
            _cloud="AzureCloud"):
        """Class Workspace constructor to load an existing Azure Machine Learning Workspace.

        :param subscription_id: The Azure subscription ID containing the workspace.
        :type subscription_id: str
        :param resource_group: The resource group containing the workspace.
        :type resource_group: str
        :param workspace_name: The workspace name.
            The name must be between 2 and 32 characters long. The first character of the name must be
            alphanumeric (letter or number), but the rest of the name may contain alphanumerics, hyphens, and
            underscores. Whitespace is not allowed.
        :type workspace_name: str
        :param auth: The authentication object. For more details, see https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication or
            azureml.core.authentication.InteractiveLoginAuthentication or azureml.core.authentication.MsiAuthentication
        :param _location: Internal use only.
        :type _location: str
        :param _disable_service_check: Internal use only.
        :type _disable_service_check: bool
        :param _workspace_id: Internal use only.
        :type _workspace_id: str
        :param sku: The parameter is present for backwards compatibility and is ignored.
        :type sku: str
        :param tags: Tags to associate with the workspace.
        :type tags: dict
        :param _cloud: Internal use only.
        :type _cloud: str
        """
        if not auth:
            auth = InteractiveLoginAuthentication()

        self._auth = auth
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        # Used to set the location in remote context.
        self._location = _location
        self._workspace_autorest_object = None

        if not _disable_service_check:
            auto_rest_workspace = _commands.get_workspace(
                auth, subscription_id, resource_group, workspace_name, _location, _cloud, _workspace_id)
            self._workspace_autorest_object = auto_rest_workspace

        self._service_context = None
        self._workspace_id_internal = _workspace_id
        self._internal_sdk_telemetry_app_insights_key = None
        self._discovery_url_internal = None
        self._sku = sku
        self._tags = tags

    @staticmethod
    def from_config(path=None, auth=None, _logger=None, _file_name=None):
        """Return a workspace object from an existing Azure Machine Learning Workspace.

        Reads workspace configuration from a file. Throws an exception if the config file can't be found.

        The method provides a simple way to reuse the same workspace across multiple Python notebooks or projects.
        Users can save the workspace Azure Resource Manager (ARM) properties using the :meth:`write_config` method,
        and use this method to load the same workspace in different Python notebooks or projects without
        retyping the workspace ARM properties.

        :param path: The path to the config file or starting directory to search.
            The parameter defaults to starting the search in the current directory.
        :type path: str
        :param auth: The authentication object. For more details, see https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication or
            azureml.core.authentication.InteractiveLoginAuthentication
        :param _logger: Allows overriding the default logger.
        :type _logger: logging.Logger
        :param _file_name: Allows overriding the config file name to search for when path is a directory path.
        :type _file_name: str
        :return: The workspace object for an existing Azure ML Workspace.
        :rtype: azureml.core.workspace.Workspace
        """
        if _logger is None:
            _logger = module_logger

        if path is None:
            path = '.'

        normalized_path = normalize_path(path)

        if os.path.isfile(normalized_path):
            found_path = normalized_path
        else:
            from azureml._base_sdk_common.common import AML_CONFIG_DIR, AZUREML_DIR, \
                LEGACY_PROJECT_FILENAME, CONFIG_FILENAME
            # Based on priority
            # Look in config dirs like .azureml, aml_config or plain directory
            # with None
            directories_to_look = [AZUREML_DIR, AML_CONFIG_DIR, None]
            if _file_name:
                files_to_look = [_file_name]
            else:
                files_to_look = [CONFIG_FILENAME, LEGACY_PROJECT_FILENAME]

            found_path = None
            for curr_dir in directories_to_look:
                for curr_file in files_to_look:
                    _logger.debug(
                        "No config file directly found, starting search from {} "
                        "directory, for {} file name to be present in "
                        "{} subdirectory".format(
                            normalized_path, curr_file, curr_dir))

                    found_path = traverse_up_path_and_find_file(
                        path=normalized_path, file_name=curr_file, directory_name=curr_dir)
                    if found_path:
                        break
                if found_path:
                    break

            if not found_path:
                raise UserErrorException(
                    "The workspace configuration file config.json, could not be found in {} or its parent "
                    "directories. Please check whether the workspace configuration file exists, or provide "
                    "the full path to the configuration file as an argument. You can download a configuration "
                    "file for your workspace, via http://ml.azure.com and clicking on the name of your "
                    "workspace in the right top.".format(
                        normalized_path))

        subscription_id, resource_group, workspace_name = project_info.get_workspace_info(
            found_path)

        _logger.info('Found the config file in: %s', found_path)
        return Workspace.get(
            workspace_name,
            auth=auth,
            subscription_id=subscription_id,
            resource_group=resource_group)

    @staticmethod
    def create(
            name,
            auth=None,
            subscription_id=None,
            resource_group=None,
            location=None,
            create_resource_group=True,
            sku='basic',
            tags=None,
            friendly_name=None,
            storage_account=None,
            key_vault=None,
            app_insights=None,
            container_registry=None,
            adb_workspace=None,
            primary_user_assigned_identity=None,
            cmk_keyvault=None,
            resource_cmk_uri=None,
            hbi_workspace=False,
            default_cpu_compute_target=None,
            default_gpu_compute_target=None,
            private_endpoint_config=None,
            private_endpoint_auto_approval=True,
            exist_ok=False,
            show_output=True,
            user_assigned_identity_for_cmk_encryption=None,
            system_datastores_auth_mode='accessKey',
            v1_legacy_mode=None):
        """Create a new Azure Machine Learning Workspace.

        Throws an exception if the workspace already exists or any of the workspace requirements are not satisfied.

        .. remarks::

            This first example requires only minimal specification, and all dependent resources as well as the
            resource group will be created automatically.

            .. code-block:: python

                from azureml.core import Workspace
                ws = Workspace.create(name='myworkspace',
                                      subscription_id='<azure-subscription-id>',
                                      resource_group='myresourcegroup',
                                      create_resource_group=True,
                                      location='eastus2')

            The following example shows how to reuse existing Azure resources utilizing the Azure resource ID format.
            The specific Azure resource IDs can be retrieved through the Azure Portal or SDK. This assumes that the
            resource group, storage account, key vault, App Insights and container registry already exist.

            .. code-block:: python

                import os
                from azureml.core import Workspace
                from azureml.core.authentication import ServicePrincipalAuthentication

                service_principal_password = os.environ.get("AZUREML_PASSWORD")

                service_principal_auth = ServicePrincipalAuthentication(
                    tenant_id="<tenant-id>",
                    username="<application-id>",
                    password=service_principal_password)

                ws = Workspace.create(name='myworkspace',
                                      auth=service_principal_auth,
                                      subscription_id='<azure-subscription-id>',
                                      resource_group='myresourcegroup',
                                      create_resource_group=False,
                                      location='eastus2',
                                      friendly_name='My workspace',
                                      storage_account='subscriptions/<azure-subscription-id>/resourcegroups/myresourcegroup/providers/microsoft.storage/storageaccounts/mystorageaccount',
                                      key_vault='subscriptions/<azure-subscription-id>/resourcegroups/myresourcegroup/providers/microsoft.keyvault/vaults/mykeyvault',
                                      app_insights='subscriptions/<azure-subscription-id>/resourcegroups/myresourcegroup/providers/microsoft.insights/components/myappinsights',
                                      container_registry='subscriptions/<azure-subscription-id>/resourcegroups/myresourcegroup/providers/microsoft.containerregistry/registries/mycontainerregistry',
                                      exist_ok=False)


        :param name: The new workspace name.
            The name must be between 2 and 32 characters long. The first character of the name must be
            alphanumeric (letter or number), but the rest of the name may contain alphanumerics, hyphens, and
            underscores. Whitespace is not allowed.
        :type name: str
        :param auth: The authentication object. For more details, see https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication or
            azureml.core.authentication.InteractiveLoginAuthentication
        :param subscription_id: The subscription ID of the containing subscription for the new workspace.
            The parameter is required if the user has access to more than one subscription.
        :type subscription_id: str
        :param resource_group: The Azure resource group that contains the workspace.
            The parameter defaults to a mutation of the workspace name.
        :type resource_group: str
        :param location: The location of the workspace.
            The parameter defaults to the resource group location.
            The location has to be a `supported
            region <https://azure.microsoft.com/global-infrastructure/services/?products=machine-learning-service>`_
            for Azure Machine Learning.
        :type location: str
        :param create_resource_group: Indicates whether to create the resource group if it doesn't exist.
        :type create_resource_group: bool
        :param sku: The parameter is present for backwards compatibility and is ignored.
        :type sku: str
        :param tags: Tags to associate with the workspace.
        :type tags: dict
        :param friendly_name: An optional friendly name for the workspace that can be displayed in the UI.
        :type friendly_name: str
        :param storage_account: An existing storage account in the Azure resource ID format.
            The storage will be used by the workspace to save run outputs, code, logs etc.
            If None, a new storage account will be created.
        :type storage_account: str
        :param key_vault: An existing key vault in the Azure resource ID format. See example code below for details
            of the Azure resource ID format.
            The key vault will be used by the workspace to store credentials added to the workspace by the users.
            If None, a new key vault will be created.
        :type key_vault: str
        :param app_insights: An existing Application Insights in the Azure resource ID format. See example code below
            for details of the Azure resource ID format.
            The Application Insights will be used by the workspace to log webservices events.
            If None, a new Application Insights will be created.
        :type app_insights: str
        :param container_registry: An existing container registry in the Azure resource ID format (see example code
            below for details of the Azure resource ID format).
            The container registry will be used by the workspace to pull and
            push both experimentation and webservices images.
            If None, a new container registry will be created only when needed and not along with workspace creation.
        :type container_registry: str
        :param adb_workspace: An existing Adb Workspace in the Azure resource ID format (see example code
            below for details of the Azure resource ID format).
            The Adb Workspace will be used to link with the workspace.
            If None, the workspace link won't happen.
        :type adb_workspace: str
        :param primary_user_assigned_identity: The resource id of the user assigned identity that used to represent
            the workspace
        :type primary_user_assigned_identity: str
        :param cmk_keyvault: The key vault containing the customer managed key in the Azure resource ID
            format:
            ``/subscriptions/<azure-subscription-id>/resourcegroups/<azure-resource-group>/providers/microsoft.keyvault/vaults/<azure-keyvault-name>``
            For example:
            '/subscriptions/d139f240-94e6-4175-87a7-954b9d27db16/resourcegroups/myresourcegroup/providers/microsoft.keyvault/vaults/mykeyvault'
            See the example code in the Remarks below for more details on the Azure resource ID format.
        :type cmk_keyvault: str
        :param resource_cmk_uri: The key URI of the customer managed key to encrypt the data at rest.
            The URI format is: ``https://<keyvault-dns-name>/keys/<key-name>/<key-version>``.
            For example, 'https://mykeyvault.vault.azure.net/keys/mykey/bc5dce6d01df49w2na7ffb11a2ee008b'.
            Refer to https://docs.microsoft.com/azure-stack/user/azure-stack-key-vault-manage-portal for steps on how
            to create a key and get its URI.
        :type resource_cmk_uri: str
        :param hbi_workspace: Specifies whether the workspace contains data of High Business Impact (HBI), i.e.,
            contains sensitive business information. This flag can be set only during workspace creation. Its value
            cannot be changed after the workspace is created. The default value is False.

            When set to True, further encryption steps are performed, and depending on the SDK component, results
            in redacted information in internally-collected telemetry. For more information, see
            `Data encryption <https://docs.microsoft.com/azure/machine-learning/
            concept-enterprise-security#data-encryption>`_.

            When this flag is set to True, one possible impact is increased difficulty troubleshooting issues.
            This could happen because some telemetry isn't sent to Microsoft and there is less visibility into
            success rates or problem types, and therefore may not be able to react as proactively when this
            flag is True. The recommendation is use the default of False for this flag unless strictly required to
            be True.
        :type hbi_workspace: bool
        :param default_cpu_compute_target: (DEPRECATED) A configuration that will be used to create a CPU compute.
            The parameter defaults to {min_nodes=0, max_nodes=2, vm_size="STANDARD_DS2_V2", vm_priority="dedicated"}
            If None, no compute will be created.
        :type default_cpu_compute_target: azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration
        :param default_gpu_compute_target:  (DEPRECATED) A configuration that will be used to create a GPU compute.
            The parameter defaults to  {min_nodes=0, max_nodes=2, vm_size="STANDARD_NC6", vm_priority="dedicated"}
            If None, no compute will be created.
        :type default_gpu_compute_target: azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration
        :param private_endpoint_config: The private endpoint configuration to create a private endpoint to
            Azure ML workspace.
        :type private_endpoint_config: azureml.core.private_endpoint.PrivateEndPointConfig
        :param private_endpoint_auto_approval: A boolean flag that denotes if the private endpoint creation should be
            auto-approved or manually-approved from Azure Private Link Center. In case of manual approval, users can
            view the pending request in Private Link portal to approve/reject the request.
        :type private_endpoint_auto_approval: bool
        :param exist_ok: Indicates whether this method succeeds if the workspace already exists. If False, this method
            fails if the workspace exists. If True, this method returns the existing workspace if it exists.
        :type exist_ok: bool
        :param show_output: Indicates whether this method will print out incremental progress.
        :type show_output: bool
        :param user_assigned_identity_for_cmk_encryption: The resource id of the user assigned identity
            that needs to be used to access the customer manage key
        :type user_assigned_identity_for_cmk_encryption: str
        :param system_datastores_auth_mode: Determines whether or not to use credentials for the system datastores of
            the workspace 'workspaceblobstore' and 'workspacefilestore'. The default value is 'accessKey', in which
            case, the workspace will create the system datastores with credentials.
            If set to 'identity', the workspace will create the system datastores with no credentials.
        :type system_datastores_auth_mode: str
        :param v1_legacy_mode: Prevent using v2 API service on public Azure Resource Manager
        :type v1_legacy_mode: bool
        :return: The workspace object.
        :rtype: azureml.core.workspace.Workspace
        :raises azureml.exceptions.WorkspaceException: Raised for problems creating the workspace.
        """
        # Checking the validity of the workspace name.
        check_valid_resource_name(name, "Workspace")
        # TODO: Checks if the workspace already exists.

        if not auth:
            auth = InteractiveLoginAuthentication()

        if not subscription_id:
            all_subscriptions = Workspace._fetch_subscriptions(auth)
            if len(all_subscriptions) > 1:
                raise WorkspaceException(
                    "You have access to more than one subscriptions. "
                    "Please specify one from this list = {}".format(all_subscriptions))
            subscription_id = all_subscriptions[0].subscription_id

        if not resource_group:
            # Resource group name should only contains lowercase alphabets.
            resource_group = Workspace._get_resource_name_from_workspace_name(
                name, "resource_group")

        if location:
            available_locations = _available_workspace_locations(
                subscription_id, auth)
            available_locations = [x.lower().replace(' ', '')
                                   for x in available_locations]
            location = location.lower().replace(' ', '')
            if location not in available_locations:
                raise WorkspaceException(
                    "Location not available for that subscription.")

        if tags is None:
            tags = {}
        # add tag in the workspace to indicate which sdk version the workspace is created from
        if tags.get("createdByToolkit") is None:
            tags["createdByToolkit"] = "sdk-v1-{}".format(VERSION)

        if default_cpu_compute_target:
            module_logger.warning(
                "'default_cpu_compute_target' is deprecated and will be removed in a future release. "
                "Use compute_targets['name'] to retrieve an existing compute target instead.")
        if default_gpu_compute_target:
            module_logger.warning(
                "'default_gpu_compute_target' is deprecated and will be removed in a future release. "
                "Use compute_targets['name'] to retrieve an existing compute target instead.")

        if private_endpoint_config is not None:
            private_endpoint_config.vnet_subscription_id = private_endpoint_config.vnet_subscription_id \
                if private_endpoint_config.vnet_subscription_id else subscription_id
            private_endpoint_config.vnet_resource_group = private_endpoint_config.vnet_resource_group \
                if private_endpoint_config.vnet_resource_group else resource_group

        return Workspace._create_legacy(
            auth,
            subscription_id,
            resource_group,
            name,
            location=location,
            create_resource_group=create_resource_group,
            sku="basic",
            tags=tags,
            friendly_name=friendly_name,
            storage_account=storage_account,
            key_vault=key_vault,
            app_insights=app_insights,
            container_registry=container_registry,
            adb_workspace=adb_workspace,
            primary_user_assigned_identity=primary_user_assigned_identity,
            cmk_keyvault=cmk_keyvault,
            resource_cmk_uri=resource_cmk_uri,
            hbi_workspace=hbi_workspace,
            default_cpu_compute_target=default_cpu_compute_target,
            default_gpu_compute_target=default_gpu_compute_target,
            private_endpoint_config=private_endpoint_config,
            private_endpoint_auto_approval=private_endpoint_auto_approval,
            exist_ok=exist_ok,
            show_output=show_output,
            user_assigned_identity_for_cmk_encryption=user_assigned_identity_for_cmk_encryption,
            system_datastores_auth_mode=system_datastores_auth_mode,
            v1_legacy_mode=v1_legacy_mode)

    @staticmethod
    def get(name, auth=None, subscription_id=None, resource_group=None, location=None, cloud="AzureCloud", id=None):
        """Return a workspace object for an existing Azure Machine Learning Workspace.

        Throws an exception if the workspace does not exist or the required fields
        do not uniquely identify a workspace.

        :param name: The name of the workspace to get.
        :type name: str
        :param auth: The authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication or
            azureml.core.authentication.InteractiveLoginAuthentication
        :param subscription_id: The subscription ID to use.
            The parameter is required if the user has access to more than one subscription.
        :type subscription_id: str
        :param resource_group: The resource group to use.
            If None, the method will search all resource groups in the subscription.
        :type resource_group: str
        :param location: The workspace location.
        :type location: str
        :param cloud: The name of the target cloud. Can be one of "AzureCloud", "AzureChinaCloud", or
            "AzureUSGovernment". If no cloud is specified "AzureCloud" is used.
        :type cloud: str
        :param id: The id of the workspace.
        :type id: str
        :return: The workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        if not auth:
            auth = InteractiveLoginAuthentication()

        return Workspace(
            subscription_id,
            resource_group,
            name,
            auth=auth,
            _location=location,
            _cloud=cloud,
            _workspace_id=id)

    @staticmethod
    def list(subscription_id, auth=None, resource_group=None):
        """List all workspaces that the user has access to within the subscription.

        The list of workspaces can be filtered based on the resource group.

        :param subscription_id: The subscription ID for which to list workspaces.
        :type subscription_id: str
        :param auth: The authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication
            or azureml.core.authentication.InteractiveLoginAuthentication
        :param resource_group: A resource group to filter the returned workspaces.
            If None, the method will list all the workspaces within the specified subscription.
        :type resource_group: str
        :return: A dictionary where the key is workspace name and the value is a list of Workspace objects.
        :rtype: dict
        """
        if not auth:
            auth = InteractiveLoginAuthentication()

        result_dict = dict()
        # All combinations of subscription_id and resource_group
        # specified/unspecified.
        if not subscription_id and not resource_group:
            all_subscriptions, _ = auth._get_all_subscription_ids()
            for subscription_tuple in all_subscriptions:
                subscription_id = subscription_tuple.subscription_id
                # Sometimes, there are subscriptions from which a user cannot read workspaces, so we just
                # ignore those while listing.
                workspaces_list = Workspace._list_legacy(
                    auth, subscription_id=subscription_id, ignore_error=True)
                Workspace._process_autorest_workspace_list(
                    auth, workspaces_list, result_dict)
        elif subscription_id and not resource_group:
            workspaces_list = Workspace._list_legacy(
                auth, subscription_id=subscription_id)
            Workspace._process_autorest_workspace_list(
                auth, workspaces_list, result_dict)
        elif not subscription_id and resource_group:
            # TODO: Need to find better ways to just query the workspace with
            # name from ARM.
            all_subscriptions = auth._get_all_subscription_ids()
            for subscription_tuple in all_subscriptions:
                subscription_id = subscription_tuple.subscription_id
                workspaces_list = Workspace._list_legacy(
                    auth,
                    subscription_id=subscription_id,
                    resource_group_name=resource_group,
                    ignore_error=True)

                Workspace._process_autorest_workspace_list(
                    auth, workspaces_list, result_dict)
        elif subscription_id and resource_group:
            workspaces_list = Workspace._list_legacy(
                auth, subscription_id=subscription_id, resource_group_name=resource_group)

            Workspace._process_autorest_workspace_list(
                auth, workspaces_list, result_dict)
        return result_dict

    def write_config(self, path=None, file_name=None):
        """Write the workspace Azure Resource Manager (ARM) properties to a config file.

        Workspace ARM properties can be loaded later using the :meth:`from_config` method. The ``path`` defaults
        to '.azureml/' in the current working directory and ``file_name`` defaults to 'config.json'.

        The method provides a simple way of reusing the same workspace across multiple Python notebooks or projects.
        Users can save the workspace ARM properties using this function,
        and use from_config to load the same workspace in different Python notebooks or projects without
        retyping the workspace ARM properties.

        :param path: User provided location to write the config.json file.
            The parameter defaults to '.azureml/' in the current working directory.
        :type path: str
        :param file_name: Name to use for the config file. The parameter defaults to config.json.
        :type file_name: str
        """
        # If path is None, use the current working directory as the path.
        if path is None:
            path = '.'

        config_dir_name = get_run_config_dir_name(path)
        normalized_config_path = normalize_path_and_join(path, config_dir_name)
        try:
            check_and_create_dir(normalized_config_path)
        except OSError as e:
            if e.errno in [errno.EPERM, errno.EACCES]:
                raise UserErrorException(
                    'You do not have permission to write the config '
                    'file to: {}\nPlease make sure you have write '
                    'permissions to the path.'.format(normalized_config_path))
            else:
                raise UserErrorException(
                    'Could not write the config file to: '
                    '{}\n{}'.format(
                        normalized_config_path, str(e)))

        if file_name is None:
            # For backcompat, we don't want both project.json or config.json to
            # be present at the same time.
            file_name = get_config_file_name(normalized_config_path)
        else:
            file_name = normalize_file_ext(file_name, 'json')

        normalized_file_path = normalize_path_and_join(
            normalized_config_path, file_name)
        scope = "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}"\
            .format(self.subscription_id,
                    self.resource_group,
                    self.name)
        try:
            project_info.add(None, scope, normalized_file_path,
                             is_config_file_path=True)
        except OSError as e:
            raise UserErrorException(
                'Could not write the config file to: '
                '{}\n{}'.format(
                    normalized_file_path, str(e)))

    @property
    def name(self):
        """Return the workspace name.

        :return: The workspace name.
        :rtype: str
        """
        return self._workspace_name

    @property
    def subscription_id(self):
        """Return the subscription ID for this workspace.

        :return: The subscription ID.
        :rtype: str
        """
        return self._subscription_id

    @property
    def resource_group(self):
        """Return the resource group name for this workspace.

        :return: The resource group name.
        :rtype: str
        """
        return self._resource_group

    @property
    def location(self):
        """Return the location of this workspace.

        :return: The location of this workspace.
        :rtype: str
        """
        has_autorest_location = (self._workspace_autorest_object is not None
                                 and self._workspace_autorest_object.location)
        if has_autorest_location:
            return self._workspace_autorest_object.location
        else:
            has_set_location = bool(self._location)
            if has_set_location:
                # Return the hard coded location in the remote context
                return self._location
            else:
                # Sets the workspace autorest object.
                self.get_details()
                return self._workspace_autorest_object.location

    @classmethod
    def _from_service_context(self, _service_context, _location):
        """Return the service context for this workspace.

        :param _service_context: A ServiceContext object.
        :type _service_context: azureml._restclient.service_context.ServiceContext
        :param _location: Internal use only.
        :type _location: str
        :return: Returns the Workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        ws = Workspace(_service_context.subscription_id,
                       _service_context.resource_group_name,
                       _service_context.workspace_name,
                       auth=_service_context.get_auth(),
                       _location=_location,
                       _disable_service_check=True,
                       _workspace_id=_service_context.workspace_id)

        ws._service_context = _service_context
        ws._discovery_url_internal = _service_context._get_discovery_url()
        return ws

    @property
    def service_context(self):
        """Return the service context for this workspace.

        :return: Returns the ServiceContext object.
        :rtype: azureml._restclient.service_context.ServiceContext
        """
        if self._service_context is not None:
            return self._service_context

        try:
            workspace_id = self._workspace_id
        except Exception:
            workspace_id = None

        try:
            discovery_url = self.discovery_url
        except Exception:
            discovery_url = None

        from azureml._restclient.service_context import ServiceContext
        self._service_context = ServiceContext(self._subscription_id,
                                               self._resource_group,
                                               self._workspace_name,
                                               workspace_id,
                                               discovery_url,
                                               self._auth)
        return self._service_context

    @property
    def sku(self):
        """Return the SKU of this workspace.

        :return: The SKU of this workspace.
        :rtype: str
        """
        has_autorest_sku = (self._workspace_autorest_object is not None
                            and self._workspace_autorest_object.sku)
        if has_autorest_sku:
            return self._workspace_autorest_object.sku
        else:
            has_set_sku = bool(self._sku)
            if has_set_sku:
                # Return the hard coded sku in the remote context
                return self._sku
            else:
                # Sets the workspace autorest object.
                self.get_details()
                return self._workspace_autorest_object.sku

    @property
    def tags(self):
        """Return the Tags of this workspace.

        :return: The Tags of this workspace.
        :rtype: dict
        """
        has_autorest_tags = (self._workspace_autorest_object is not None
                             and self._workspace_autorest_object.tags)
        if has_autorest_tags:
            return self._workspace_autorest_object.tags
        else:
            has_set_tags = bool(self._tags)
            if has_set_tags:
                # Return the hard coded tags in the remote context
                return self._tags
            else:
                # Sets the workspace autorest object.
                self.get_details()
                return self._workspace_autorest_object.tags

    @property
    def _sdk_telemetry_app_insights_key(self):
        """Return the internal SDK telemetry key.

        :return: The internal SDK telemetry key.
        :rtype: str
        """
        if self._internal_sdk_telemetry_app_insights_key is None:
            if self._workspace_autorest_object is None:
                # Attempt to set the workspace autorest object.
                self.get_details()
            if self._workspace_autorest_object:
                self._internal_sdk_telemetry_app_insights_key = \
                    self._workspace_autorest_object.sdk_telemetry_app_insights_key
        return self._internal_sdk_telemetry_app_insights_key

    @property
    def discovery_url(self):
        """Return the discovery URL of this workspace.

        :return: The discovery URL of this workspace.
        :rtype: str
        """
        has_autorest_discovery_url = (self._workspace_autorest_object is not None
                                      and self._workspace_autorest_object.discovery_url)
        if has_autorest_discovery_url:
            return self._workspace_autorest_object.discovery_url
        else:
            discovery_url = self._discovery_url_internal
            if discovery_url is not None:
                return discovery_url
            else:
                # Sets the workspace autorest object.
                self.get_details()
                return self._workspace_autorest_object.discovery_url

    def update_dependencies(self, container_registry=None, force=False):
        """Update existing the associated resources for workspace in the following cases.

            a) When a user accidently deletes an existing associated resource and would like to
            update it with a new one without having to recreate the whole workspace.
            b) When a user has an existing associated resource and wants to replace the current one
            that is associated with the workspace.
            c) When an associated resource hasn’t been created yet and they want to use an existing one
            that they already have (only applies to container registry).

        :param container_registry: ARM Id for the container registry.
        :type container_registry: str
        :param force: If force updating dependent resources without prompted confirmation.
        :type force: bool
        :rtype: azureml.core.workspace.Workspace
        """
        if not container_registry:
            print("No resource is specified to be updated")
            return
        if not force and container_registry is not None:
            give_warning(
                "By updating the workspace with a new container registry, "
                + "existing environments will need to be pre-built in the new registry. "
                + "Any runs currently in progress may fail. "
                + "Any models using existing environments will need to be redeployed. "
                "Press 'y' to confirm")
            yes_set = ['yes', 'y', 'ye', '']
            choice = input().lower()
            if choice not in yes_set:
                print("customer disagreed to update container registry")
                return

        workspace_autorest = _commands.update_workspace(
            auth=self._auth,
            resource_group_name=self._resource_group,
            workspace_name=self._workspace_name,
            subscription_id=self._subscription_id,
            container_registry=container_registry)
        return convert_dict_keys_to_camel_case(workspace_autorest.as_dict())

    def delete(self, delete_dependent_resources=False, no_wait=False):
        """Delete the Azure Machine Learning Workspace associated resources.

        :param delete_dependent_resources: Whether to delete
            resources associated with the workspace, i.e., container registry, storage account, key vault, and
            application insights. The default is False. Set to True to delete these resources.
        :type delete_dependent_resources: bool
        :param no_wait: Whether to wait for the workspace deletion to complete.
        :type no_wait: bool
        :return: None if successful; otherwise, throws an error.
        :rtype: None
        """
        _commands.delete_workspace(
            self._auth,
            self._resource_group,
            self._workspace_name,
            self._subscription_id,
            delete_dependent_resources,
            no_wait=no_wait)

    def get_details(self):
        """Return the details of the workspace.

        .. remarks::

            The returned dictionary contains the following key-value pairs.

            * `id`: URI pointing to this workspace resource, containing
              subscription ID, resource group, and workspace name.
            * `name`: The name of this workspace.
            * `location`: The workspace region.
            * `type`: A URI of the format "{providerName}/workspaces".
            * `tags`: Not currently used.
            * `workspaceid`: The ID of this workspace.
            * `description`: Not currently used.
            * `friendlyName`: A friendly name for the workspace displayed in the UI.
            * `creationTime`: Time this workspace was created, in ISO8601 format.
            * `containerRegistry`: The workspace container registry used to pull and
              push both experimentation and webservices images.
            * `keyVault`: The workspace key vault used to store credentials added to the workspace by the users.
            * `applicationInsights`: The Application Insights will be used by the workspace to log webservices events.
            * `identityPrincipalId`:
            * `identityTenantId`
            * `identityType`
            * `storageAccount`: The storage will be used by the workspace to save run outputs, code, logs, etc.
            * `sku`: The workspace SKU (also referred as edition).
              The parameter is present for backwards compatibility and is ignored.
            * `resourceCmkUri`: The key URI of the customer managed key to encrypt the data at rest. Refer to
              https://docs.microsoft.com/en-us/azure-stack/user/azure-stack-key-vault-manage-portal?view=azs-1910 for
              steps on how to create a key and get its URI.
            * `hbiWorkspace`: Specifies if the customer data is of high business impact.
            * `imageBuildCompute`: The compute target for image build.
            * `systemDatastoresAuthMode`: Determines whether or not to use credentials for the system datastores
              of the workspace 'workspaceblobstore' and 'workspacefilestore'. The default value is 'accessKey',
              in which case, the workspace will create the system datastores with credentials.
              If set to 'identity', the workspace will create the system datastores with no credentials.

            For more information on these key-value pairs, see :meth:`create <create>`.

        :return: Workspace details in dictionary format.
        :rtype: dict[str, str]
        """
        # TODO: Need to change the return type.
        self._workspace_autorest_object = _commands.show_workspace(
            self._auth, self._resource_group, self._workspace_name,
            self._subscription_id)
        # workspace_autorest is an object of
        # azureml._base_sdk_common.workspace.models.workspace.Workspace
        return convert_dict_keys_to_camel_case(
            self._workspace_autorest_object.as_dict())

    def get_mlflow_tracking_uri(self, _with_auth=False):
        """Get the MLflow tracking URI for the workspace.

        MLflow (https://mlflow.org/) is an open-source platform for tracking machine learning experiments
        and managing models. You can use MLflow logging APIs with Azure Machine Learning so that metrics,
        models and artifacts are logged to your Azure Machine Learning workspace.

        .. remarks::

            Use the following sample to configure MLflow tracking to send data to the Azure ML Workspace:

            .. code-block:: python

                import mlflow
                from azureml.core import Workspace
                workspace = Workspace.from_config()
                mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

        :param _with_auth: (DEPRECATED) Add auth info to tracking URI.
        :type _with_auth: bool
        :return: The MLflow-compatible tracking URI.
        :rtype: str
        """
        if _with_auth:
            module_logger.warning(
                "'_with_auth' is deprecated and will be removed in a future release. ")

        try:
            from azureml.mlflow import get_mlflow_tracking_uri
            return get_mlflow_tracking_uri(self)
        except ImportError as e:
            error_msg = "azureml.mlflow could not be imported. "\
                        "Please ensure that 'azureml-mlflow' has been installed in the current python environment."
            raise UserErrorException(error_msg, inner_exception=e)

    @property
    def compute_targets(self):
        """List all compute targets in the workspace.

        :return: A dictionary with key as compute target name and value as :class:`azureml.core.ComputeTarget`
            object.
        :rtype: dict[str, azureml.core.ComputeTarget]
        """
        return {
            compute_target.name: compute_target for compute_target in ComputeTarget.list(self)}

    def get_default_compute_target(self, type):
        """Get the default compute target for the workspace.

        :param type: The type of compute. Possible values are 'CPU' or 'GPU'.
        :type type: str

        :return: The default compute target for given compute type.
        :rtype: azureml.core.ComputeTarget
        """
        module_logger.warning(
            "'get_default_compute_target' is deprecated and will be removed in a future release. "
            "Use compute_targets['name'] to retrieve an existing compute target instead.")
        if type.lower() == "cpu":
            thing = Workspace.DEFAULT_CPU_CLUSTER_NAME
        elif type.lower() == "gpu":
            thing = Workspace.DEFAULT_GPU_CLUSTER_NAME
        else:
            raise ValueError(
                "Invalid compute type. Accepted values are CPU and GPU")

        return self.compute_targets.get(thing, None)

    @property
    def datastores(self):
        """List all datastores in the workspace. This operation does not return credentials of the datastores.

        :return: A dictionary with key as datastore name and value as :class:`azureml.core.datastore.Datastore`
            object.
        :rtype: dict[str, azureml.core.Datastore]
        """
        return {
            datastore.name: datastore for datastore in _DatastoreClient.list(self)}

    @property
    def datasets(self):
        """List all datasets in the workspace.

        :return: A dictionary with key as dataset name and value as :class:`azureml.core.dataset.Dataset`
            object.
        :rtype: dict[str, azureml.core.Dataset]
        """
        return Dataset.get_all(self)

    def get_default_datastore(self):
        """Get the default datastore for the workspace.

        :return: The default datastore.
        :rtype: azureml.core.datastore.Datastore
        """
        return _DatastoreClient.get_default(self)

    def set_default_datastore(self, name):
        """Set the default datastore for the workspace.

        :param name: The name of the :class:`azureml.core.datastore.Datastore` to set as default.
        :type name: str
        """
        _DatastoreClient.set_default(self, name)

    def get_default_keyvault(self):
        """Get the default key vault object for the workspace.

        :return: The KeyVault object associated with the workspace.
        :rtype: azureml.core.keyvault.Keyvault
        """
        return Keyvault(self)

    def add_private_endpoint(
            self,
            private_endpoint_config,
            private_endpoint_auto_approval=True,
            location=None,
            show_output=True,
            tags=None):
        """Add a private endpoint to the workspace.

        :param private_endpoint_config: The private endpoint configuration to create a private endpoint to workspace.
        :type private_endpoint_config: azureml.core.private_endpoint.PrivateEndPointConfig
        :param private_endpoint_auto_approval: A boolean flag that denotes if the private endpoint creation should be
            auto-approved or manually-approved from Azure Private Link Center. In case of manual approval, users can
            view the pending request in Private Link portal to approve/reject the request.
        :type private_endpoint_auto_approval: bool
        :param location: Location of the private endpoint, default is the workspace location
        :type location: string
        :param show_output: Flag for showing the progress of workspace creation
        :type show_output: bool
        :param tags: Tags to associate with the workspace.
        :type tags: dict

        :return: The PrivateEndPoint object created.
        :rtype: azureml.core.private_endpoint.PrivateEndPoint
        """
        private_endpoint_config.vnet_subscription_id = private_endpoint_config.vnet_subscription_id \
            if private_endpoint_config.vnet_subscription_id else self.subscription_id
        private_endpoint_config.vnet_resource_group = private_endpoint_config.vnet_resource_group \
            if private_endpoint_config.vnet_resource_group else self.resource_group
        if not location:
            location = self.location

        return _commands.add_workspace_private_endpoint(
            self._auth,
            self.subscription_id,
            self.resource_group,
            self.name,
            location,
            private_endpoint_config,
            private_endpoint_auto_approval,
            tags,
            show_output)

    def delete_private_endpoint_connection(
            self, private_endpoint_connection_name):
        """Delete the private endpoint connection to the workspace.

        :param private_endpoint_connection_name: The unique name of private endpoint connection under the workspace
        :type private_endpoint_connection_name: str
        """
        return _commands.delete_workspace_private_endpoint_connection(
            self._auth,
            self.subscription_id,
            self.resource_group,
            self.name,
            private_endpoint_connection_name)

    def list_keys(self):
        """List keys for the current workspace.

        :return:
        :rtype: object
        """
        # TODO: Need to change the return type.
        result = _commands.list_workspace_keys(
            self._auth,
            self._subscription_id,
            self._resource_group,
            self._workspace_name)
        return convert_dict_keys_to_camel_case(result.as_dict())

    def diagnose_workspace(self, diagnose_parameters):
        """Diagnose workspace setup issues.

        :param diagnose_parameters: The parameter of diagnosing workspace health
        :type diagnose_parameters: ~_restclient.models.DiagnoseWorkspaceParameters
        :return: An instance of AzureOperationPoller that returns
         DiagnoseResponseResult
        :rtype:
         ~msrestazure.azure_operation.AzureOperationPoller[~_restclient.models.DiagnoseResponseResult]
        """
        diagnose_result = _commands.diagnose_workspace(
            self._auth,
            self._subscription_id,
            self._resource_group,
            self._workspace_name,
            diagnose_parameters=diagnose_parameters)
        return convert_dict_keys_to_camel_case(diagnose_result.as_dict())

    def set_connection(self, name, category, target, authType, value):
        """Add or update a connection under the workspace.

        :param name: The unique name of connection under the workspace
        :type name: str
        :param category: The category of this connection
        :type category: str
        :param target: the target this connection connects to
        :type target: str
        :param authType: the authorization type of this connection
        :type authType: str
        :param value: the json format serialization string of the connection details
        :type value: str
        """
        result = _commands.create_or_update_workspace_connection(
            self._auth,
            self.subscription_id,
            self.resource_group,
            self.name,
            name,
            category,
            target,
            authType,
            value)
        return convert_dict_keys_to_camel_case(result.as_dict())

    def list_connections(self, category=None, target=None):
        """List connections under this workspace.

        :param type: The type of this connection that will be filtered on
        :type type: str
        :param target: the target of this connection that will be filtered on
        :type target: str
        """
        list_result = _commands.list_workspace_connections(
            self._auth,
            self.subscription_id,
            self.resource_group,
            self.name,
            category,
            target)
        return "[%s]" % ', '.join(map(lambda result: str(
            convert_dict_keys_to_camel_case(result.as_dict())), list_result))

    def get_connection(self, name):
        """Get a connection of the workspace.

        :param name: The unique name of connection under the workspace
        :type name: str
        """
        result = _commands.get_workspace_connection(
            self._auth, self.subscription_id, self.resource_group, self.name, name)
        return convert_dict_keys_to_camel_case(result.as_dict())

    def delete_connection(self, name):
        """Delete a connection of the workspace.

        :param name: The unique name of connection under the workspace
        :type name: str
        """
        _commands.delete_workspace_connection(
            self._auth,
            self.subscription_id,
            self.resource_group,
            self.name,
            name)

    @property
    def private_endpoints(self):
        """List all private endpoint of the workspace.

        :return: A dict of PrivateEndPoint objects associated with the workspace. The key is private endpoint name.
        :rtype: dict[str, azureml.core.private_endpoint.PrivateEndPoint]
        """
        return _commands.get_workspace_private_endpoints(
            self._auth, self.subscription_id, self.resource_group, self.name)

    @property
    def experiments(self):
        """List all experiments in the workspace.

        :return: A dictionary with key as experiment name and value as :class:`azureml.core.Experiment` object.
        :rtype: dict[str, azureml.core.Experiment]
        """
        from azureml.core.experiment import Experiment as _Experiment
        return {
            experiment.name: experiment for experiment in _Experiment.list(self)}

    @property
    def environments(self):
        """List all environments in the workspace.

        :return: A dictionary with key as environment name and value as :class:`azureml.core.Environment` object.
        :rtype: dict[str, azureml.core.Environment]
        """
        from azureml.core.environment import Environment
        return Environment.list(self)

    @property
    def images(self):
        """Return the list of images in the workspace.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem interacting with
        model management service.

        :return: A dictionary with key as image name and value as :class:`azureml.core.image.Image` object.
        :rtype: dict[str, azureml.core.Image]
        :raises azureml.exceptions.WebserviceException: There was a problem interacting with the model management
            service.
        """
        images = Image.list(self)
        result = {}
        for name in set([image.name for image in images]):
            result[name] = max([image for image in images if image.name == name],
                               key=attrgetter('version'))

        return result

    @property
    def models(self):
        """Return a list of model in the workspace.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem interacting with
        model management service.

        :return: A dictionary of model with key as model name and value as :class:`azureml.core.model.Model` object.
        :rtype: dict[str, azureml.core.Model]
        :raises azureml.exceptions.WebserviceException: There was a problem interacting with the model management
            service.
        """
        return {model.name: model for model in Model.list(self, latest=True)}

    @property
    def webservices(self):
        """Return a list of webservices in the workspace.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem returning the list.

        :return: A list of webservices in the workspace.
        :rtype: dict[str, azureml.core.Webservice]
        :raises azureml.exceptions.WebserviceException: There was a problem returning the list.
        """
        return {
            webservice.name: webservice for webservice in Webservice.list(self)}

    @property
    def linked_services(self):
        """List all linked services in the workspace.

        :return: A dictionary where key is a linked service name and value is a :class:`azureml.core.LinkedService`
            object.
        :rtype: dict[str, azureml.core.LinkedService]
        """
        return {
            linked_service.name: linked_service for linked_service in LinkedService.list(self)}

    def get_run(self, run_id):
        """Return the run with the specified run_id in the workspace.

        :param run_id: The run ID.
        :type run_id: string
        :return: The submitted run.
        :rtype: azureml.core.run.Run
        """
        from azureml.core.run import Run
        return Run.get(self, run_id)

    @staticmethod
    def _fetch_subscriptions(auth):
        """Get all subscriptions a user has access to.

        If a user has access to only one subscription than that is  returned, otherwise an exception is thrown.

        :param auth: The authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
        :type auth: azureml.core.authentication.AbstractAuthentication
        :return: A subscription id
        :rtype: str
        """
        all_subscriptions = auth._get_all_subscription_ids()
        if len(all_subscriptions) == 0:
            raise WorkspaceException(
                "You don't have access to any subscriptions. "
                "Workspace operation failed.")

        return all_subscriptions

    @property
    def _workspace_id(self):
        """Return workspace id that is uniquely identifies a workspace.

        :return: Returns the workspace id that uniquely identifies a workspace.
        :rtype: str
        """
        if self._workspace_id_internal is None:
            if self._workspace_autorest_object is None:
                # Sets the workspace autorest object.
                self.get_details()
            self._workspace_id_internal = self._workspace_autorest_object.workspaceid
        return self._workspace_id_internal

    @staticmethod
    def _get_or_create(
            name,
            auth=None,
            subscription_id=None,
            resource_group=None,
            location=None,
            sku='basic',
            friendly_name=None,
            storage_account=None,
            key_vault=None,
            app_insights=None,
            container_registry=None,
            adb_workspace=None,
            create_resource_group=True,
            cmk_keyvault=None,
            resource_cmk_uri=None,
            hbi_workspace=False,
            default_cpu_compute_target=None,
            default_gpu_compute_target=None,
            user_assigned_identity_for_cmk_encryption=None,
            system_datastores_auth_mode='accessKey',
            v1_legacy_mode=None):
        """Get or create a workspace if it doesn't exist.

        Throws an exception if the required fields are not specified.

        :param name: The workspace name.
        :type name: str
        :param auth: Authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
        :type auth: azureml.core.authentication.AbstractAuthentication
        :param subscription_id: The subscription id to use.
        :type subscription_id: str
        :param resource_group: The resource group to use.
        :type resource_group: str
        :param location: The workspace location in-case azureml SDK has to create a workspace.
        :type location: str
        :param sku: The workspace SKU. The parameter is present for backwards compatibility and is ignored.
        :type sku: str
        :param friendly_name: The friendly name of the workspace.
        :type friendly_name: str
        :param storage_account: The storage account to use for this workspace.
        :type storage_account: str
        :param app_insights: The app insights to use for this workspace.
        :type app_insights: str
        :param key_vault: The keyvault to use for this workspace.
        :type key_vault: str
        :param container_registry: The container registry to use for this workspace.
        :type container_registry: str
        :param adb_workspace: The adb workspace to link with this workspace.
        :type adb_workspace: str
        :param create_resource_group: Flag to create resource group or not.
        :type create_resource_group: bool
        :param cmk_keyvault: The key vault containing the customer managed key in the Azure resource ID
            format.
        :type cmk_keyvault: str
        :param resource_cmk_uri: The key URI of the customer managed key to encrypt the data at rest. Refer to
            https://docs.microsoft.com/azure-stack/user/azure-stack-key-vault-manage-portal for steps on how to
            create a key and get its URI.
        :type resource_cmk_uri: str
        :param hbi_workspace: Specifies whether the customer data is of High Business Impact(HBI), i.e., contains
            sensitive business information. The default value is False. When set to True, downstream services
            will selectively disable logging. This flag can be set only during workspace creation. Its value
            cannot be changed after the workspace is created. The default value is False.
        :type hbi_workspace: bool
        :param default_cpu_compute_target: (DEPRECATED) A configuration that will be used to create a CPU compute.
            The parameter defaults to {min_nodes=0, max_nodes=2, vm_size="STANDARD_DS2_V2", vm_priority="dedicated"}
            If None, no compute will be created.
        :type default_cpu_compute_target: azureml.core.AmlCompute.AmlComputeProvisioningConfiguration
        :param default_gpu_compute_target: (DEPRECATED) A configuration that will be used to create a GPU compute.
            The parameter defaults to  {min_nodes=0, max_nodes=2, vm_size="STANDARD_NC6", vm_priority="dedicated"}
            If None, no compute will be created.
        :type default_gpu_compute_target: azureml.core.AmlCompute.AmlComputeProvisioningConfiguration
        :param user_assigned_identity_for_cmk_encryption: The resource id of the user assigned identity
            that needs to be used to access the customer managed key
        :type user_assigned_identity_for_cmk_encryption: str
        :param system_datastores_auth_mode: Determines whether or not to use credentials for the system datastores of
            the workspace 'workspaceblobstore' and 'workspacefilestore'. The default value is 'accessKey', in which
            case, the workspace will create the system datastores with credentials.  If set to 'identity',
            the workspace will create the system datastores with no credentials.
        :type system_datastores_auth_mode: str
        :param v1_legacy_mode: Prevent using v2 API service on public Azure Resource Manager
        :type v1_legacy_mode: bool
        :return: The workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        # Checking the validity of the workspace name.
        check_valid_resource_name(name, "Workspace")

        try:
            return Workspace.get(
                name,
                auth=auth,
                subscription_id=subscription_id,
                resource_group=resource_group)

        # Added ProjectSystemException because internally we are throwing
        # ProjectSystemException when a workspace is not found.
        except (WorkspaceException, ProjectSystemException) as e:
            # Distinguishing the case when multiple workspaces with the same
            # name already exists.
            if isinstance(e, WorkspaceException) and e.found_multiple:
                # In this case, we throw the error.
                raise e
            else:
                # Workspace doesn't exist, so we create it.
                if default_cpu_compute_target:
                    module_logger.warning(
                        "'default_cpu_compute_target' is deprecated and will be removed in a future release. "
                        "Use compute_targets['name'] to retrieve an existing compute target instead.")
                if default_gpu_compute_target:
                    module_logger.warning(
                        "'default_gpu_compute_target' is deprecated and will be removed in a future release. "
                        "Use compute_targets['name'] to retrieve an existing compute target instead.")
                return Workspace.create(
                    name,
                    auth=auth,
                    subscription_id=subscription_id,
                    resource_group=resource_group,
                    location=location,
                    create_resource_group=create_resource_group,
                    sku="basic",
                    friendly_name=friendly_name,
                    storage_account=storage_account,
                    key_vault=key_vault,
                    app_insights=app_insights,
                    container_registry=container_registry,
                    adb_workspace=adb_workspace,
                    cmk_keyvault=cmk_keyvault,
                    resource_cmk_uri=resource_cmk_uri,
                    hbi_workspace=hbi_workspace,
                    default_cpu_compute_target=default_cpu_compute_target,
                    default_gpu_compute_target=default_gpu_compute_target,
                    user_assigned_identity_for_cmk_encryption=user_assigned_identity_for_cmk_encryption,
                    system_datastores_auth_mode=system_datastores_auth_mode,
                    v1_legacy_mode=v1_legacy_mode)

    @staticmethod
    def _list_legacy(auth, subscription_id=None,
                     resource_group_name=None, ignore_error=False):
        """List workspaces.

        :param auth: Authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
        :type auth: azureml.core.authentication.AbstractAuthentication
        :param subscription_id:
        :type subscription_id: str
        :param resource_group_name:
        :type resource_group_name: str
        :param ignore_error: ignore_error=True, ignores any errors and returns an empty list.
        :type ignore_error: bool
        :return: list of object of azureml._base_sdk_common.workspace.models.workspace.Workspace
        :rtype: list
        """
        try:
            # A list of object of
            # azureml._base_sdk_common.workspace.models.workspace.Workspace
            workspace_autorest_list = _commands.list_workspace(
                auth, subscription_id=subscription_id, resource_group_name=resource_group_name)
            return workspace_autorest_list
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise e

    @staticmethod
    def _create_legacy(
            auth,
            subscription_id,
            resource_group_name,
            workspace_name,
            location=None,
            create_resource_group=None,
            sku="basic",
            tags=None,
            friendly_name=None,
            storage_account=None,
            key_vault=None,
            app_insights=None,
            container_registry=None,
            adb_workspace=None,
            primary_user_assigned_identity=None,
            cmk_keyvault=None,
            resource_cmk_uri=None,
            hbi_workspace=False,
            default_cpu_compute_target=None,
            default_gpu_compute_target=None,
            private_endpoint_config=None,
            private_endpoint_auto_approval=None,
            exist_ok=False,
            show_output=True,
            user_assigned_identity_for_cmk_encryption=None,
            system_datastores_auth_mode="accessKey",
            v1_legacy_mode=None):
        """Create a workspace.

        :param auth: Authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
        :type auth: azureml.core.authentication.AbstractAuthentication
        :param subscription_id: The subscription ID to use.
        :type subscription_id: str
        :param resource_group_name: The resource group to use.
        :type resource_group_name: str
        :param workspace_name: The workspace name to use.
        :type workspace_name: str
        :param location: The workspace location in-case azureml SDK has to create a workspace.
        :type location: str
        :param create_resource_group: Flag to create resource group or not.
        :type create_resource_group: bool
        :param sku: The SKU name. The parameter is present for backwards compatibility and is ignored.
        :type sku: str
        :param tags: Tags to associate with the workspace.
        :type tags: dict
        :param friendly_name: The friendly name of the workspace.
        :type friendly_name: str
        :param storage_account: The storage account to use for this workspace.
        :type storage_account: str
        :param key_vault: The keyvault to use for this workspace.
        :type key_vault: str
        :param app_insights: The app insights to use for this workspace.
        :type app_insights: str
        :param container_registry: The container registry to use for this workspace.
        :type container_registry: str
        :param adb_workspace: The adb workspace to link with this workspace.
        :type adb_workspace: str
        :param primary_user_assigned_identity: The resource id of the user assigned identity that used to represent
            the workspace
        :type primary_user_assigned_identity: str
        :param cmk_keyvault: The key vault containing the customer managed key in the Azure resource ID
            format.
        :type cmk_keyvault: str
        :param resource_cmk_uri: The key URI of the customer managed key to encrypt the data at rest. Refer to
            https://docs.microsoft.com/azure-stack/user/azure-stack-key-vault-manage-portal for steps on how to
            create a key and get its URI.
        :type resource_cmk_uri: str
        :param hbi_workspace: Specifies whether the customer data is of High Business Impact(HBI), i.e., contains
            sensitive business information. The default value is False. When set to True, downstream services
            will selectively disable logging. This flag can be set only during workspace creation. Its value
            cannot be changed after the workspace is created. The default value is False.
        :type hbi_workspace: bool
        :param exist_ok: Indicates whether this method succeeds if the workspace already exists. If False, this method
            fails if the workspace exists. If True, this method returns the existing workspace if it exists.
        :type exist_ok: bool
        :param show_output: Flag for showing the progress of workspace creation
        :type show_output: bool
        :param user_assigned_identity_for_cmk_encryption: The resource id of the user assigned identity
            that needs to be used to access the customer manage key
        :type user_assigned_identity_for_cmk_encryption: str
        :param system_datastores_auth_mode: Determines whether or not to use credentials for the system datastores of
            the workspace 'workspaceblobstore' and 'workspacefilestore'. The default value is 'accessKey', in which
            case, the workspace will create the system datastores with credentials.  If set to 'identity',
            the workspace will create the system datastores with no credentials.
        :type system_datastores_auth_mode: str
        :param v1_legacy_mode: Prevent using v2 API service on public Azure Resource Manager
        :type v1_legacy_mode: bool
        :return: The created workspace
        :rtype: azureml.core.workspace.Workspace
        """
        workspace_object_autorest = _commands.create_or_update_workspace(
            auth, resource_group_name, workspace_name, subscription_id, location=location,
            create_resource_group=create_resource_group,
            sku=sku,
            tags=tags,
            friendly_name=friendly_name,
            storage_account=storage_account,
            key_vault=key_vault,
            app_insights=app_insights,
            containerRegistry=container_registry,
            adb_workspace=adb_workspace,
            primary_user_assigned_identity=primary_user_assigned_identity,
            cmk_keyvault=cmk_keyvault,
            resource_cmk_uri=resource_cmk_uri,
            hbi_workspace=hbi_workspace,
            default_cpu_compute_target=default_cpu_compute_target,
            default_gpu_compute_target=default_gpu_compute_target,
            private_endpoint_config=private_endpoint_config,
            private_endpoint_auto_approval=private_endpoint_auto_approval,
            exist_ok=exist_ok, show_output=show_output,
            user_assigned_identity_for_cmk_encryption=user_assigned_identity_for_cmk_encryption,
            system_datastores_auth_mode=system_datastores_auth_mode,
            v1_legacy_mode=v1_legacy_mode)
        if not workspace_object_autorest:
            raise WorkspaceException("Couldn't create the workspace.")

        # Disabling service check as we just created the workspace.
        workspace_object = Workspace(
            subscription_id,
            resource_group_name,
            workspace_name,
            auth=auth,
            _disable_service_check=True,
            _location=location,
            sku=sku,
            tags=tags)

        # workspace_object_autorest is an object of class
        # azureml._base_sdk_common.workspace.models.workspace.Workspace
        workspace_object._workspace_autorest_object = workspace_object_autorest
        return workspace_object

    def _to_dict(self):
        """Serialize this workspace information into a dictionary.

        :return:
        :rtype: dict
        """
        result_dict = dict()
        result_dict["subscriptionId"] = self._subscription_id
        result_dict["resourceGroup"] = self._resource_group
        result_dict["workspaceName"] = self._workspace_name
        return result_dict

    @staticmethod
    def _get_workspace_from_autorest_workspace(auth, auto_rest_workspace):
        """Return workspace.

        :param auth: Authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
        :type auth: azureml.core.authentication.AbstractAuthentication
        :param auto_rest_workspace:
        :type auto_rest_workspace: azureml._base_sdk_common.workspace.models.workspace.Workspace
        :return:
        :rtype: azureml.core.workspace.Workspace
        """
        workspace_scope = Workspace._get_scope_details(auto_rest_workspace.id)
        # Disabling the service check as we just got autorest objects from the
        # service.
        workspace_object = Workspace(
            workspace_scope.subscription_id,
            workspace_scope.resource_group,
            workspace_scope.workspace_name,
            auth=auth,
            _disable_service_check=True)
        # Set workspace autorest object only if it is of type Workspace to avoid setting
        # it when ARM returns a GenericResource
        if isinstance(auto_rest_workspace, _ModelsWorkspace):
            workspace_object._workspace_autorest_object = auto_rest_workspace
        return workspace_object

    @staticmethod
    def _process_autorest_workspace_list(
            auth, workspace_autorest_list, result_dict):
        """Process a list of workspaces returned by the autorest client and adds those to result_dict.

        Implementation details: key is a  workspace name, value is a list of workspace objects.

        :param auth: Authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
        :type auth: azureml.core.authentication.AbstractAuthentication
        :param workspace_autorest_list: list of object of azureml._base_sdk_common.workspace.models.workspace.Workspace
        :type workspace_autorest_list: list
        :param result_dict: A dict to add workspaces.
        :type result_dict: dict
        :return:
        :rtype: None
        """
        if not workspace_autorest_list:
            return

        for workspace_autorest_object in workspace_autorest_list:
            workspace_object = Workspace._get_workspace_from_autorest_workspace(
                auth, workspace_autorest_object)

            if workspace_object.name in result_dict:
                result_dict[workspace_object.name].append(workspace_object)
            else:
                item_list = list()
                item_list.append(workspace_object)
                result_dict[workspace_object.name] = item_list

    @staticmethod
    def _get_scope_details(workspace_arm_scope):
        """Parse the arm scope of a workspace and returns WorkspaceScopeInfo tuple.

        :param workspace_arm_scope:
        :type workspace_arm_scope:
        :return:
        :rtype: _WorkspaceScopeInfo
        """
        subscription_id = re.search(
            r'/subscriptions/([^/]+)', workspace_arm_scope).group(1)
        resource_group = re.search(
            r'/resourceGroups/([^/]+)', workspace_arm_scope).group(1)
        workspace_name = re.search(
            r'/workspaces/([^/]+)', workspace_arm_scope).group(1)
        workspace_scope_info = _WorkspaceScopeInfo(
            subscription_id, resource_group, workspace_name)
        return workspace_scope_info

    @property
    def _auth_object(self):
        """Get authentication object.

        :return: Returns the auth object.
        :rtype: azureml.core.authentication.AbstractAuthentication
        """
        return self._auth

    def update(self, friendly_name=None, description=None,
               tags=None, image_build_compute=None,
               service_managed_resources_settings=None,
               primary_user_assigned_identity=None,
               allow_public_access_when_behind_vnet=None,
               v1_legacy_mode=None):
        """Update friendly name, description, tags, image build compute and other settings associated with a workspace.

        :param friendly_name: A friendly name for the workspace that can be displayed in the UI.
        :type friendly_name: str
        :param description: A description of the workspace.
        :type description: str
        :param tags: Tags to associate with the workspace.
        :type tags: dict
        :param image_build_compute: The compute name for the image build.
        :type image_build_compute: str
        :param service_managed_resources_settings: The service managed resources settings.
        :type service_managed_resources_settings:
            azureml._base_sdk_common.workspace.models.ServiceManagedResourcesSettings
        :param primary_user_assigned_identity: The user assigned identity resource
            id that represents the workspace identity.
        :type primary_user_assigned_identity: str
        :param allow_public_access_when_behind_vnet: Allow public access to private link workspace.
        :type allow_public_access_when_behind_vnet: bool
        :param v1_legacy_mode: Prevent using v2 API service on public Azure Resource Manager
        :type v1_legacy_mode: bool
        :return: A dictionary of updated information.
        :rtype: dict[str, str]
        """
        workspace_autorest = _commands.update_workspace(
            self._auth,
            self._resource_group,
            self._workspace_name,
            self._subscription_id,
            friendly_name=friendly_name,
            description=description,
            tags=tags,
            image_build_compute=image_build_compute,
            service_managed_resources_settings=service_managed_resources_settings,
            primary_user_assigned_identity=primary_user_assigned_identity,
            allow_public_access_when_behind_vnet=allow_public_access_when_behind_vnet,
            v1_legacy_mode=v1_legacy_mode)
        return convert_dict_keys_to_camel_case(workspace_autorest.as_dict())

    def _get_create_status_dict(self):
        """Return the create status in dict format.

        :return:
        :rtype: dict
        """
        # TODO: Will need to remove this function. Adding this to pass tests
        # for now.
        return_dict = convert_dict_keys_to_camel_case(
            self._workspace_autorest_object.as_dict())
        if "resourceGroup" not in return_dict:
            return_dict["resourceGroup"] = self._resource_group
        return return_dict

    @staticmethod
    def _rand_gen(num_chars):
        """Generate random string.

        :param num_chars:
        :type num_chars: int
        :return: Returns randomly generated string.
        :rtype: str
        """
        return ''.join(choice(ascii_lowercase) for i in range(num_chars))

    @staticmethod
    def _get_resource_name_from_workspace_name(workspace_name, resource_type):
        """Return the resource name.

        :param workspace_name:
        :type workspace_name: str
        :param resource_type:
        :type resource_type: str
        :return: Returns the resource name.
        :rtype: str
        """
        alphabets_str = ""
        for char in workspace_name.lower():
            if char.isalpha():
                alphabets_str = alphabets_str + char

        # Account name should be in lower case.
        # So, getting alphabets from the workspace name
        resource_name = alphabets_str[:8] + \
            resource_type + Workspace._rand_gen(20)

        # Storage account names in azure can only be 24 characters longs.
        if len(resource_name) > 24:
            return resource_name[:24]
        else:
            return resource_name

    def _initialize_folder(self, experiment_name, directory="."):
        """Initialize a folder with all files needed for an experiment.

        If the path specified by directory doesn't exist then the SDK creates those directories.

        :param experiment_name: The experiment name.
        :type experiment_name: str
        :param directory: The directory path.
        :type directory: str
        :return: The project object.
        :rtype: azureml.core.project.Project
        """
        # Keeping the import here to prevent the cyclic dependency between
        # Workspace and Project.
        from azureml._project.project import Project
        return Project.attach(self, experiment_name, directory=directory)

    def sync_keys(self, no_wait=False):
        """Triggers the workspace to immediately synchronize keys.

        If keys for any resource in the workspace are changed, it can take around an hour for them to automatically
        be updated. This function enables keys to be updated upon request. An example scenario is needing immediate
        access to storage after regenerating storage keys.

        :param no_wait: Whether to wait for the workspace sync keys to complete.
        :type no_wait: bool
        :return: None if successful; otherwise, throws an error.
        :rtype: None
        """
        return self._sync_keys(no_wait)

    def _sync_keys(self, no_wait):
        """Sync keys for the current workspace.

        :return:
        :rtype: object
        """
        # TODO: Need to change the return type.
        return _commands.workspace_sync_keys(
            self._auth,
            self._resource_group,
            self._workspace_name,
            self._subscription_id,
            no_wait)

    def _share(self, user, role):
        """Share the current workspace.

        :param user:
        :type user: str
        :param role:
        :type role: str
        :return:
        :rtype: object
        """
        return _commands.share_workspace(
            self._auth,
            self._resource_group,
            self._workspace_name,
            self._subscription_id,
            user,
            role)

    def __repr__(self):
        """Produce an unambiguous 'formal' string representation of this workspace.

        return: A string representation of the workspace.
        rtype: str
        """
        return "Workspace.create(name='%s', subscription_id='%s', resource_group='%s')" \
               % (self._workspace_name, self._subscription_id, self._resource_group)

    @staticmethod
    def _workspace_parameters_input():
        """
        Turn on interactive mode for the user to enter their subscription, resource group and workspace information.

        :return: Tuple of information entered by the user
        :rtype: namedtuple
        """
        ws_entries = collections.namedtuple(
            'ws_entries', [
                'subscription_id', 'resource_group', 'workspace_name', 'workspace_region'])

        print("-- No *.aml_env* file could be found - Let's create one --\n-- Please provide (without quotes):")
        subscription_id = input("  > Your subscription ID: ")
        resource_group = input("  > Your resource group: ")
        workspace_name = input("  > The name of your workspace: ")
        workspace_region = input(
            "  > The region of your workspace\n(full list available at "
            "https://azure.microsoft.com/en-us/global-infrastructure/geographies/): ")

        all_entries = ws_entries(
            subscription_id.replace(
                " ", ""), resource_group.replace(
                " ", ""), workspace_name.replace(
                " ", ""), workspace_region.replace(
                " ", ""))
        # Removes any extra leading, ending, intermediate space

        if '' in all_entries:
            raise ValueError("Please provide non empty values")

        return all_entries

    @staticmethod
    def setup():
        """
        Create a new workspace or retrieve an existing workspace.

        :return: A Workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        # Load the workspace
        module_logger.warning(
            "setup() is now deprecated. Instead, please use create() to create a new workspace, "
            "or get()/from_config() to retrieve an existing one")
        try:
            # From a configuration file
            ws = Workspace.from_config()
        except UserErrorException:
            ws_parameters = Workspace._workspace_parameters_input()
            try:
                # Or directly from Azure
                ws = Workspace(subscription_id=ws_parameters.subscription_id,
                               resource_group=ws_parameters.resource_group,
                               workspace_name=ws_parameters.workspace_name)
                # And generate a local configuration file
                ws.write_config()
                print(
                    "Workspace *{}* configuration was successfully retrieved".format(ws.name))
            except (UserErrorException, ProjectSystemException):
                # If the workspace does not exist, asks user to choose between re-entering the information
                # and creating a brand new workspace
                print(
                    ">>> We could not find the workspace you are looking for.\nPlease enter:")
                choice = input(
                    "1: To create a new workspace with the parameters above\n"
                    "Any other key: To start over\n")
                if choice == "1":
                    print(
                        "Creating workspace *{}* from scratch ...".format(ws_parameters.workspace_name))
                    ws = Workspace.create(
                        name=ws_parameters.workspace_name,
                        subscription_id=ws_parameters.subscription_id,
                        resource_group=ws_parameters.resource_group,
                        location=ws_parameters.workspace_region,
                        create_resource_group=True,
                        sku='basic')
                    ws.write_config()
                    print(
                        "Workspace *{}* has been successfully created.".format(ws.name))
                else:
                    ws = Workspace.setup()
        return ws


def _available_workspace_locations(subscription_id, auth=None):
    """List available locations/azure regions where an azureml workspace can be created.

    :param subscription_id: The subscription ID.
    :type subscription_id: str
    :param auth: Authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
    :type auth: azureml.core.authentication.AbstractAuthentication
    :return: The list of azure regions where an azureml workspace can be created.
    :rtype: list[str]
    """
    if not auth:
        auth = InteractiveLoginAuthentication()
    return _commands.available_workspace_locations(auth, subscription_id)
