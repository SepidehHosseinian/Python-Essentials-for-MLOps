# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import logging
import os
import time

from azureml._restclient.clientbase import ClientBase
from azure.core.exceptions import HttpResponseError
from msrest.authentication import DomainCredentials

from azureml._base_sdk_common.common import create_role_assignment
from azureml._base_sdk_common.common import resource_client_factory, give_warning
from azureml._base_sdk_common.common import resource_error_handling
from azureml._base_sdk_common.workspace import AzureMachineLearningWorkspaces
from azureml._base_sdk_common.workspace.models import (ErrorResponseWrapperException,
                                                       WorkspaceUpdateParameters,
                                                       WorkspaceConnectionProps,
                                                       WorkspaceConnectionDto)
from azureml._base_sdk_common.workspace.operations import WorkspacesOperations, WorkspaceConnectionsOperations, PrivateEndpointConnectionsOperations
from azureml._base_sdk_common.workspace.operations.private_link_resources_operations import PrivateLinkResourcesOperations
from azureml._base_sdk_common.common import get_http_exception_response_string
from azureml._project.project_engine import ProjectEngineClient
from azureml._workspace._utils import (
    delete_kv_armId,
    delete_storage_armId,
    delete_insights_armId
)
from azure.mgmt.resource import ResourceManagementClient
from azureml.exceptions import ProjectSystemException, WorkspaceException
from msrest.exceptions import ClientRequestError, HttpOperationError
from msrestazure.azure_exceptions import CloudError
from azureml.core.private_endpoint import PrivateEndPoint
from azure.core.exceptions import ResourceNotFoundError

module_logger = logging.getLogger(__name__)

AML_DISCOVERY_API_HOST_ENV_NAME = "AML_DISCOVERY_API_HOST"

# TODO: These keys should be moved to a base_sdk_common place.
# Resource type keys
PROJECT = "Project"
WORKSPACE = "Workspace"

# Project info keys
RESOURCE_GROUP_KEY = "resourceGroups"
WORKSPACE_KEY = "workspaces"
PROJECT_KEY = "projects"
SUBSCRIPTION_KEY = "subscriptions"

# Synapse env variables
AZURE_SERVICE = "AZURE_SERVICE"
SYNAPSE = "Microsoft.ProjectArcadia"

_KNOWN_AML_DISCOVERY_API_HOST = {
    "azurecloud": "api.azureml.ms",
    "azurechinacloud": "api.ml.azure.cn",
    "azuregermancloud": "api.ml.azure.de",
    "azureusgovernment": "api.ml.azure.us"
}

def get_aml_discovery_host(cloud_name: str):
    cloud_discovery_api_host_env_name = cloud_name.upper() + "_" + AML_DISCOVERY_API_HOST_ENV_NAME
    aml_discovery_host = os.getenv(cloud_discovery_api_host_env_name)
    if aml_discovery_host:
        return aml_discovery_host
    aml_discovery_host = os.getenv(AML_DISCOVERY_API_HOST_ENV_NAME)
    if (aml_discovery_host):
        return aml_discovery_host
    aml_discovery_host = _KNOWN_AML_DISCOVERY_API_HOST.get(cloud_name.lower(), None)
    if (aml_discovery_host):
        return aml_discovery_host

    raise ValueError("AML discovery api host not found for cloud {}. Set one of following environment variables, {} or {}".format(cloud_name, cloud_discovery_api_host_env_name, AML_DISCOVERY_API_HOST_ENV_NAME))

# TODO: After passing project_object, we might be able to remove some of
# function arguments in each function.
""" Modules """


def get_project_info(auth, path, exception_bypass=False):
    """

    :param auth:
    :param path:
    :param exception_bypass:
    :return:
    """
    """Get project information from path"""
    # Get project id from Engine
    project_scope = ProjectEngineClient.get_project_scope_by_path(path)

    if not project_scope:
        if exception_bypass:
            return None
        message = "No cache found for current project, try providing resource group"
        message += " and workspace arguments"
        raise ProjectSystemException(message)
    else:
        content = project_scope.split('/')
        keys = content[1::2]
        values = content[2::2]
        return dict(zip(keys, values))


def attach_project(workspace_object, cloud_project_name, project_path="."):
    """
    Attaches a machine learning project to a local directory, specified by project_path.
    :param workspace_object:
    :type workspace_object: azureml.core.workspace.Workspace
    :param cloud_project_name:
    :param project_path:
    :return: A dict of project attach related information.
    :rtype: dict
    """
    from azureml._restclient.experiment_client import ExperimentClient
    experiment_name = cloud_project_name
    project_path = os.path.abspath(project_path)

    #  Start temporary fix while project service is being deprecated
    workspace_service_object, _ = _get_or_create_workspace(
        workspace_object._auth_object, workspace_object.subscription_id, workspace_object.resource_group,
        workspace_object.name, workspace_object.location)

    experiment_client = ExperimentClient(
    workspace_object.service_context, experiment_name)

    scope = "{}/{}/{}".format(experiment_client.get_workspace_uri_path(),
                              PROJECT_KEY, experiment_name)
    # End temporary fix

    # Send appropriate information
    project_engine_client = ProjectEngineClient(workspace_object._auth_object)
    reply_dict = project_engine_client.attach_project(experiment_name,
                                                      project_path, scope, workspace_object.compute_targets)

    # Adding location information
    reply_dict["experimentName"] = experiment_name
    return reply_dict


def _get_or_create_workspace(
    auth,
    subscription_id,
    resource_group_name,
    workspace_name,
     location):
    """
    Gets or creates a workspace.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :param resource_group_name:
    :type resource_group_name: str
    :param workspace_name:
    :type workspace_name: str
    :param location:
    :type location: str
    :return: Returns the workspace object and a bool indicating if the workspace was newly created.
    :rtype: azureml._base_sdk_common.workspace.models.workspace.Workspace, bool
    """
    from azureml._base_sdk_common.workspace.models import Workspace
    newly_created = False
    try:
        # try to get the workspace first
        workspace_object = WorkspacesOperations.get(
            auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
            resource_group_name, workspace_name)
    except HttpOperationError as response_exception:
        if response_exception.response.status_code == 404:
            # if no workspace, create the workspace
            params = Workspace(location=location, friendly_name=workspace_name)
            newly_created = True
            workspace_object = WorkspacesOperations.create_or_update(
                auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
                resource_group_name, workspace_name, params)
        else:
            raise ProjectSystemException(
    get_http_exception_response_string(
        response_exception.response))

    return workspace_object, newly_created


def detach_project(project_path="."):
    """
    Deletes the project.json from a project.
    Raises an exception if the file deletion fails.
    :param project_path: The project path.
    :type project_path: str
    :return:
    """
    from azureml._project import project_info
    project_info.delete_project_json(os.path.abspath(project_path))


def list_project(auth, subscription_id=None, resource_group_name=None, workspace_name=None,
                 local=False):
    """
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param local:
    :param subscription_id:
    :return:
    """
    """List Projects"""
    from azureml._restclient.workspace_client import WorkspaceClient
    # If local flag is specified, list all local projects
    if local:
        project_engine_client = ProjectEngineClient(auth)
        return project_engine_client.get_local_projects()

    # Subscription id cannot be none while listing cloud projects.
    if not subscription_id:
        raise ProjectSystemException(
            "Subscription id cannot be none while listing cloud projects.")

    try:
        if resource_group_name and workspace_name:
            workspace_client = WorkspaceClient(None,
                                               auth,
                                               subscription_id,
                                               resource_group_name,
                                               workspace_name)
            experiments = workspace_client.list_experiments()

            return [experiment.as_dict() for experiment in experiments]
        else:
            raise ProjectSystemException(
                "Please specify resource_group_name and workspace_name ")

    except ErrorResponseWrapperException as response_exception:
        resource_error_handling(response_exception, PROJECT + "s")


def basic_project_info(project_info):
    basic_dict = {
        "id": project_info["id"],
        "name": project_info["name"]
    }

    if "summary" in project_info:
        basic_dict["summary"] = project_info["summary"]

    if "description" in project_info:
        basic_dict["description"] = project_info["description"]

    if "action_hyperlink" in project_info:
        basic_dict["github_link"] = project_info["action_hyperlink"]["address"]

    return basic_dict


def create_or_update_workspace(auth, resource_group_name, workspace_name, subscription_id,
                     location=None, create_resource_group=None, sku='basic', tags=None,
                     friendly_name=None,
                     storage_account=None, key_vault=None, app_insights=None, containerRegistry=None, adb_workspace=None,
                     primary_user_assigned_identity=None,
                     cmk_keyvault=None, resource_cmk_uri=None,
                     hbi_workspace=False,
                     default_cpu_compute_target=None, default_gpu_compute_target=None,
                     private_endpoint_config=None, private_endpoint_auto_approval=None,
                     exist_ok=False, show_output=True,
                     user_assigned_identity_for_cmk_encryption=None,
                     system_datastores_auth_mode='accesskey',
                     v1_legacy_mode=None,
                     is_update_dependent_resources=False,
                     api_version="2021-01-01"):
    """
    Create or update workspace
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param subscription_id:
    :type subscription_id: str
    :param location:
    :param sku: The workspace SKU - basic or enterprise
    :type sku: str
    :param friendly_name:
    :type container_registry: str
    :param adb_workspace: the adb workspace associated with aml workspace
    :type adb_workspace: str
    :param primary_user_assigned_identity: The resource id of the user assigned identity that used to represent
            the workspace
    :type primary_user_assigned_identity: str
    :param cmk_keyvault: The key vault containing the customer managed key
     in the Azure resource ID format.
    :type cmk_keyvault: str
    :param resource_cmk_uri: The key URI of the customer managed key to encrypt the data at rest.
    :type resource_cmk_uri: str
    :param hbi_workspace: Specifies whether the customer data is of High Business Impact(HBI), i.e., contains
            sensitive business information.
    :type hbi_workspace: bool
    :param default_cpu_compute_target: A configuration that will be used to create a CPU compute.
        If None, no compute will be created.
    :type default_cpu_compute_target: azureml.core.AmlCompute.AmlComputeProvisioningConfiguration
    :param default_gpu_compute_target: A configuration that will be used to create a GPU compute.
        If None, no compute will be created.
    :param user_assigned_identity_for_cmk_encryption: The resource id of the user assigned identity that needs to be used to access the
        customer manage key
    :type user_assigned_identity_for_cmk_encryption: str
    :return:
    :rtype: azureml._base_sdk_common.workspace.models.workspace.Workspace
    :param system_datastores_auth_mode: Determines whether or not to use credentials for the system datastores
        of the workspace 'workspaceblobstore' and 'workspacefilestore'.
        The default value is 'accessKey', in which case,
        the workspace will create the system datastores with credentials.
        If set to 'identity', the workspace will create the system datastores with no credentials.
    :type system_datastores_auth_mode: str
    :param v1_legacy_mode: Prevent using v2 API service on public Azure Resource Manager
    :type v1_legacy_mode: bool
    """

    vnet = _validate_private_endpoint(auth, private_endpoint_config, location)
    vnet_location = None
    if vnet:
        vnet_location = vnet.location

    resource_management_client = resource_client_factory(auth, subscription_id)
    found = True
    try:
        resource_management_client.resource_groups.get(resource_group_name)
    except HttpResponseError as e:
        if e.status_code == 404:
            # Resource group not found case.
            found = False
        else:
            from azureml._base_sdk_common.common import get_http_exception_response_string
            raise WorkspaceException(
    get_http_exception_response_string(
        e.response))

    if not found:
        if not location:
            raise WorkspaceException("Resource group was not found and location was not provided. "
                                     "Provide location with --location to create a new resource group.")
        else:
            rg_location = location

        if create_resource_group is None:
            # Flag was not set, we need to prompt user for creation of resource
            # group
            give_warning("UserWarning: The resource group doesn't exist or was not provided. "
                         "AzureML SDK can create a resource group={} in location={} "
                         "using subscription={}. Press 'y' to confirm".format(resource_group_name,
                                                                              rg_location,
                                                                              subscription_id))

            yes_set = ['yes', 'y', 'ye', '']
            choice = input().lower()

            if choice in yes_set:
                create_resource_group = True
            else:
                raise WorkspaceException("Resource group was not found and "
                                         "confirmation prompt was denied.")

        if create_resource_group:
            # Create the required resource group, give the user details about
            # it.
            give_warning("UserWarning: The resource group doesn't exist or was not provided. "
                         "AzureML SDK is creating a resource group={} in location={} "
                         "using subscription={}.".format(resource_group_name, rg_location, subscription_id))
            from azure.mgmt.resource.resources.models import ResourceGroup
            rg_tags = {"creationTime": str(time.time()),
                       "creationSource": "azureml-sdk"}
            # Adding location as keyworded argument for compatibility with azure-mgmt-resource 1.2.*
            # and azure-mgmt-resource 2.0.0
            resource_group_properties = ResourceGroup(
                location=rg_location, tags=rg_tags)
            resource_management_client.resource_groups.create_or_update(resource_group_name,
                                                                        resource_group_properties)
        else:
            # Create flag was set to false, this path is not possible through
            # the cli
            raise WorkspaceException("Resource group was not found.")

    from azureml._workspace.custom import ml_workspace_create_resources
    return ml_workspace_create_resources(
        auth, auth._get_service_client(
    AzureMachineLearningWorkspaces, subscription_id).workspaces,
        resource_group_name,
        workspace_name,
        location,
        vnet_location=vnet_location,
        subscription_id=subscription_id,
        storage_account=storage_account,
        key_vault=key_vault,
        app_insights=app_insights,
        containerRegistry=containerRegistry,
        adbWorkspace=adb_workspace,
        friendly_name=friendly_name,
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
        sku=sku,
        tags=tags,
        user_assigned_identity_for_cmk_encryption=user_assigned_identity_for_cmk_encryption,
        system_datastores_auth_mode=system_datastores_auth_mode,
        v1_legacy_mode=v1_legacy_mode,
        is_update_dependent_resources=is_update_dependent_resources,
        api_version=api_version)


def available_workspace_locations(auth, subscription_id):
    """Lists available locations/azure regions where an azureml workspace can be created.
    :param auth: Authentication object.
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id: The subscription id.
    :type subscription_id: str
    :return: The list of azure regions where an azureml workspace can be created.
    :rtype: list[str]
    """
    response = auth._get_service_client(ResourceManagementClient, subscription_id).providers.get(
        "Microsoft.MachineLearningServices")
    for resource_type in response.resource_types:
        # There are multiple resource types like workspaces, 'workspaces/computes', 'operations' and some more.
        # All return the same set of locations.
        if resource_type.resource_type == "workspaces":
            return resource_type.locations


def get_workspace(
    auth,
    subscription_id,
    resource_group_name,
    workspace_name,
    location=None,
    cloud="azurecloud",
     workspace_id=None):
    """
    Returns the workspace object from the service.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :param resource_group_name:
    :type resource_group_name: str
    :param workspace_name:
    :type workspace_name: str
    :return: The service workspace object.
    :rtype: azureml._base_sdk_common.workspace.models.workspace.Workspace
    """
    try:
        if(os.environ.get(AZURE_SERVICE) == SYNAPSE and location is not None and cloud is not None and workspace_id is not None):
            # This is in synapse environment.
            # Use dataplane get workspace calls in synapse env, as ARM calls are blocked from synapse DEP env.
            return _get_workspace_dataplane(auth, subscription_id, resource_group_name, workspace_name, location, 
            cloud, workspace_id)
        else:
            try:
                workspaces = auth._get_service_client(
                    AzureMachineLearningWorkspaces,
                    subscription_id).workspaces
                return WorkspacesOperations.get(
                    workspaces,
                    resource_group_name,
                    workspace_name)
            except ClientRequestError:
                # this most likely is because of failure to talk to arm.
                # Provide error message to user if they are in synapse enviroment.
                # Additional params will enable aml dataplane get workspace call. 
                if(os.environ.get(AZURE_SERVICE) == SYNAPSE and (location is None or cloud is None or workspace_id is None)):
                    raise ValueError('Specify location, cloud, id parameters in get workspace.')
    except ErrorResponseWrapperException as response_exception:
        module_logger.error(
            "get_workspace error using subscription_id={}, resource_group_name={}, workspace_name={}".format(
                subscription_id, resource_group_name, workspace_name
            ))
        resource_error_handling(response_exception, WORKSPACE)


def _get_workspace_dataplane(
    auth, subscription_id, resource_group_name, workspace_name, location=None,
    cloud="azurecloud", workspace_id=None):

    domain = get_aml_discovery_host(cloud)

    if location.lower() == "master" or location.lower() == "centraluseuap":
        location = "master"
        domain = "api.azureml-test.ms"

    discovery_url = "https://{}.workspace.{}.{}/discovery/workspaces/{}".format(
        workspace_id, location, domain, workspace_id)
    from azureml._base_sdk_common.service_discovery import ServiceDiscovery
    service_discovery = ServiceDiscovery(auth)
    base_url = service_discovery.discover_services_uris(discovery_url)["api"]
    workspaces = auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id, base_url=base_url, is_check_subscription=False).workspaces

    return WorkspacesOperations.get(
    workspaces,
    resource_group_name,
    workspace_name,
     is_dataplane=True)


def list_workspace(auth, subscription_id, resource_group_name=None):
    """
    List Workspaces
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :param resource_group_name:
    :return: A list of objects of azureml._base_sdk_common.workspace.models.workspace.Workspace
    :rtype: list[azureml._base_sdk_common.workspace.models.workspace.Workspace]
    """
    try:
        if resource_group_name:
            list_object = WorkspacesOperations.list_by_resource_group(
                auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
                resource_group_name)
            workspace_list = list_object.value
            next_link = list_object.next_link

            while next_link:
                list_object = WorkspacesOperations.list_by_resource_group(
                    auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
                    resource_group_name, next_link=next_link)
                workspace_list += list_object.value
                next_link = list_object.next_link
            return workspace_list
        else:
            # Ignore any params and list all workspaces user has access to
            # optionally scoped by resource group
            from azureml._workspace.custom import ml_workspace_list
            return ml_workspace_list(
    auth, subscription_id, resource_group_name=resource_group_name)
    except ErrorResponseWrapperException as response_exception:
        resource_error_handling(response_exception, WORKSPACE + "s")


def delete_workspace(auth, resource_group_name, workspace_name, subscription_id,
                     delete_dependent_resources, no_wait):
    """
    Delete a workspace.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param subscription_id:
    :type subscription_id: str
    :return:
    """
    from azureml._workspace.custom import ml_workspace_delete
    try:
        if delete_dependent_resources:
            # get the workspace object first
            workspace = WorkspacesOperations.get(
                auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
                resource_group_name,
                workspace_name)

            delete_storage_armId(auth, workspace.storage_account)
            delete_kv_armId(auth, workspace.key_vault)
            delete_insights_armId(auth, workspace.application_insights)

        return ml_workspace_delete(
    auth,
    subscription_id,
    resource_group_name,
    workspace_name,
     no_wait)
    except ErrorResponseWrapperException as response_exception:
        resource_error_handling(response_exception, WORKSPACE)


def workspace_sync_keys(
    auth,
    resource_group_name,
    workspace_name,
    subscription_id,
     no_wait):
    """
    Sync keys associated with this workspace.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param subscription_id:
    :type subscription_id: str
    :param no_wait: Whether to wait for the workspace sync keys to complete.
    :type no_wait: bool
    :return:
    """
    try:
        azure_poller = WorkspacesOperations.sync_keys(
            auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
            resource_group_name, workspace_name)
        if no_wait:
            return
        azure_poller.wait()
    except ErrorResponseWrapperException as response_exception:
        resource_error_handling(response_exception, WORKSPACE)


def share_workspace(
    auth,
    resource_group_name,
    workspace_name,
    subscription_id,
    user,
     role):
    """
    Share this workspace with another user.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param subscription_id:
    :param user:
    :param role:
    :type subscription_id: str
    :return:
    """
    scope = '/subscriptions/' + subscription_id + '/resourceGroups/' + resource_group_name + \
        '/providers/Microsoft.MachineLearningServices/workspaces/' + workspace_name
    resolve_assignee = False
    if user.find('@') >= 0:  # user principal
        resolve_assignee = True
    create_role_assignment(
        auth,
        role,
        assignee=user,
        resource_group_name=None,
        scope=scope,
        resolve_assignee=resolve_assignee)


def update_workspace(auth, resource_group_name, workspace_name, subscription_id,
                    friendly_name=None, description=None, tags=None, image_build_compute=None,
                    service_managed_resources_settings=None,
                    primary_user_assigned_identity=None,
                    allow_public_access_when_behind_vnet=None,
                    v1_legacy_mode=None,
                    container_registry=None):
    """
    Update workspace
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param friendly_name:
    :param description: str
    :param subscription_id:
    :type subscription_id: str
    :param tags:
    :param image_build_compute: str
    :param primary_user_assigned_identity: The user assigned identity resource
        id that represents the workspace identity.
    :type primary_user_assigned_identity: str
    :param allow_public_access_when_behind_vnet: bool
    :param v1_legacy_mode: bool
    :param container_registry str
    :return:
    """
    try:
        from azureml._workspace.custom import ml_workspace_update
        return ml_workspace_update(auth._get_service_client(
            AzureMachineLearningWorkspaces, subscription_id).workspaces,
            resource_group_name,
            workspace_name,
            friendly_name,
            description,
            tags,
            image_build_compute,
            service_managed_resources_settings,
            primary_user_assigned_identity=primary_user_assigned_identity,
            allow_public_access_when_behind_vnet=allow_public_access_when_behind_vnet,
            v1_legacy_mode=v1_legacy_mode,
            container_registry=container_registry)
    except ErrorResponseWrapperException as response_exception:
        resource_error_handling(response_exception, WORKSPACE)


def show_workspace(auth, resource_group_name=None, workspace_name=None,
                   subscription_id=None):
    """
    Show Workspace
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :param workspace_name:
    :param subscription_id:
    :type subscription_id: str
    :return:
    """
    try:
        if not (resource_group_name and workspace_name and subscription_id):
            project_info = get_project_info(auth, os.getcwd())
            resource_group_name = project_info[RESOURCE_GROUP_KEY]
            workspace_name = project_info[WORKSPACE_KEY]
            subscription_id = project_info[SUBSCRIPTION_KEY]

        return WorkspacesOperations.get(auth._get_service_client(
            AzureMachineLearningWorkspaces, subscription_id).workspaces,
            resource_group_name,
            workspace_name,
            subscription_id=subscription_id)
    except ErrorResponseWrapperException as response_exception:
        resource_error_handling(response_exception, WORKSPACE)


def list_workspace_keys(
    auth,
    subscription_id,
    resource_group_name,
     workspace_name):
    workspace_keys = WorkspacesOperations.list_keys(
        auth._get_service_client(
    AzureMachineLearningWorkspaces,
     subscription_id).workspaces,
        resource_group_name, workspace_name)
    return workspace_keys


def diagnose_workspace(
    auth,
    subscription_id,
    resource_group_name,
    workspace_name,
     diagnose_parameters):
    azure_poller = WorkspacesOperations.diagnose(
    auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspaces,
    subscription_id,
    resource_group_name,
    workspace_name,
    body=diagnose_parameters)
    diagnose_result = azure_poller.result()
    return diagnose_result

def create_or_update_workspace_connection(auth, subscription_id, resource_group_name, workspace_name,
                                   connection_name, connection_category, connection_target, connection_authtype, connection_value):
    workspace_connection_object=WorkspaceConnectionsOperations.create(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspace_connections,
        resource_group_name, workspace_name, connection_name,
        WorkspaceConnectionDto(connection_name, WorkspaceConnectionProps(connection_category, connection_target, connection_authtype, connection_value)))
    return workspace_connection_object

def list_workspace_connections(auth, subscription_id, resource_group_name, workspace_name,
                                   connection_category, connection_target):
    workspace_connection_objects = WorkspaceConnectionsOperations.list(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspace_connections,
        resource_group_name, workspace_name, target=connection_target, category=connection_category)
    return workspace_connection_objects

def get_workspace_connection(auth, subscription_id, resource_group_name, workspace_name,
                                   connection_name):
    workspace_connection_object = WorkspaceConnectionsOperations.get(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspace_connections,
        resource_group_name, workspace_name, connection_name)
    return workspace_connection_object

def delete_workspace_connection(auth, subscription_id, resource_group_name, workspace_name,
                                   connection_name):
    WorkspaceConnectionsOperations.delete(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspace_connections,
        resource_group_name, workspace_name, connection_name)
    return

def list_private_link_resources(auth, subscription_id, resource_group_name, workspace_name):
    private_link_resources_object = PrivateLinkResourcesOperations.list_by_workspace(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).private_link_resources,
        resource_group_name, workspace_name)
    return private_link_resources_object

def add_workspace_private_endpoint(auth, subscription_id, resource_group_name, workspace_name, location,
                                   private_endpoint_config, private_endpoint_auto_approval, tags, show_output):
    _validate_private_endpoint(auth, private_endpoint_config, location)
    # When users are adding PE to existing workspace show a warning.
    module_logger.warning("While enabling private link on an existing workspace, please note that if there are "
                          "compute targets associated with the workspace, those targets will not work if they are not "
                          "behind the same virtual network as the workspace private endpoint.")
    from azureml._workspace.custom import create_workspace_private_endpoint
    create_workspace_private_endpoint(auth, resource_group_name, location, subscription_id, workspace_name,
                                      private_endpoint_config, private_endpoint_auto_approval, tags, show_output)
    workspace_object = WorkspacesOperations.get(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspaces,
        resource_group_name, workspace_name)
    private_endpoint = None
    if workspace_object.private_endpoint_connections is not None:
        private_endpoint = [pe for pe in workspace_object.private_endpoint_connections if
                            pe.properties.private_endpoint.id.endswith(private_endpoint_config.name)]
    if private_endpoint:
        return PrivateEndPoint(private_endpoint[0].id, private_endpoint[0].properties.private_endpoint.id)
    else:
        raise WorkspaceException("Specified private endpoint {} not found in workspace {}".format(
            private_endpoint_config.name, workspace_name))


def get_workspace_private_endpoints(auth, subscription_id, resource_group_name, workspace_name):
    # There is no PE API that lists all PEs for a given workspace.
    # So relying on get workspace that also returns the PE information.
    workspace_object = WorkspacesOperations.get(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspaces,
        resource_group_name, workspace_name)
    pe_connections = {}

    if workspace_object.private_endpoint_connections is not None:
        pe_connections = {pe.properties.private_endpoint.id.split("/")[-1]:
                              PrivateEndPoint(pe.id, pe.properties.private_endpoint.id) for pe in
                          workspace_object.private_endpoint_connections}
    return pe_connections


def delete_workspace_private_endpoint_connection(auth, subscription_id, resource_group_name, workspace_name, private_endpoint_connection_name):
    return PrivateEndpointConnectionsOperations.delete(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).private_endpoint_connections,
        resource_group_name,
        workspace_name,
        private_endpoint_connection_name)


def get_private_endpoint_details(auth, private_endpoint_resource_id):
    # resource id format = "/subscriptions/ID/resourceGroups/rg/providers/Microsoft.Network/privateEndpoint/pe_name"
    subscription_id = private_endpoint_resource_id.split("/")[2]
    return WorkspacesOperations.get_private_endpoint_connection(
        auth._get_service_client(AzureMachineLearningWorkspaces, subscription_id).workspaces,
        private_endpoint_resource_id)


def _validate_private_endpoint(auth, private_endpoint_config, location):
    # If user provided PE config information, the vnet has to be pre-created.
    # This is a requirement for workspace private link.
    # Workspace create request will be failed if vnet is specified and not found.
    if private_endpoint_config:
        from azureml._vendor.azure_network import NetworkManagementClient
        try:
            network_client = auth._get_service_client(NetworkManagementClient, private_endpoint_config.vnet_subscription_id)
            vnet = network_client.virtual_networks.get(private_endpoint_config.vnet_resource_group,
                                                       private_endpoint_config.vnet_name)
            if vnet.location != location:
                # raise WorkspaceException("The specified vnet for private endpoint and workspace should be "
                #                          "in the same region.")
                give_warning("The specified virtual network for the private endpoint must be in the same region "
                             "as the workspace to create AML compute clusters and compute instances. "
                             "If you want to use either of these compute targets, "
                             "please use a virtual network that is in the same region as the workspace.")
            network_client.subnets.get(private_endpoint_config.vnet_resource_group, private_endpoint_config.vnet_name,
                                       private_endpoint_config.vnet_subnet_name)
            return vnet
        except (CloudError, ResourceNotFoundError) as e:
            if e.status_code == 404:
                raise WorkspaceException("Specified vnet and subnet was not found in subscription={}, "
                                         "resource group= {} "
                                         "Please make sure the vnet exists to create workspace private endpoint."
                                         .format(private_endpoint_config.vnet_subscription_id,
                                                 private_endpoint_config.vnet_resource_group))


def _get_workspace_dp_from_base_url(
    auth, base_url, workspace_name, subscription_id, resource_group):
    try:
        workspaces = auth._get_service_client(
            AzureMachineLearningWorkspaces, subscription_id, base_url=base_url, is_check_subscription=False).workspaces
        autorest_ws = ClientBase._execute_func(
            WorkspacesOperations.get, workspaces, resource_group, workspace_name, is_dataplane=True)
        return autorest_ws
    except Exception as ex:
        raise ValueError("Could not get workspace {} in subscription {} and resource group "
                         "{} due to following error {} "
                         .format(workspace_name, subscription_id, resource_group, ex))
