# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli.workspace.workspace_subgroup import WorkspaceSubGroup
from azureml._cli.cli_command import command
from azureml._cli import argument

from azureml.core.workspace import Workspace
from azureml.core.private_endpoint import PrivateEndPointConfig
from azureml._base_sdk_common.workspace.models.diagnose_workspace_parameters import DiagnoseWorkspaceParameters
from azureml._base_sdk_common.workspace.models.diagnose_request_properties import DiagnoseRequestProperties
from azureml.core import VERSION


def create_workspace(
        workspace_name,
        resource_group_name=None,
        location=None,
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
        create_resource_group=None,
        exist_ok=False,
        sku='basic',
        tags=None,
        pe_name=None,
        pe_vnet_name=None,
        pe_subnet_name=None,
        pe_subscription_id=None,
        pe_resource_group=None,
        pe_auto_approval=None,
        user_assigned_identity_for_cmk_encryption=None,
        system_datastores_auth_mode='accessKey',
        v1_legacy_mode=None):

    from azureml._base_sdk_common.cli_wrapper._common import get_cli_specific_auth, get_default_subscription_id, \
        get_resource_group_or_default_name

    auth = get_cli_specific_auth()
    default_subscription_id = get_default_subscription_id(auth)

    # resource group can be None, as we create on user's behalf.
    resource_group_name = get_resource_group_or_default_name(
        resource_group_name, auth=auth)

    private_endpoint_config = None
    if pe_name is not None:
        private_endpoint_config = PrivateEndPointConfig(
            pe_name, pe_vnet_name, pe_subnet_name, pe_subscription_id, pe_resource_group)

    tags = get_tags_dict(tags)
    # add tag in the workspace to indicate which cli version the workspace is created from
    if tags.get("createdByToolkit") is None:
        tags["createdByToolkit"] = "cli-v1-{}".format(VERSION)

    workspace_object = Workspace.create(
        workspace_name,
        auth=auth,
        subscription_id=default_subscription_id,
        resource_group=resource_group_name,
        location=location,
        create_resource_group=create_resource_group,
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
        exist_ok=exist_ok,
        sku=sku,
        tags=tags,
        private_endpoint_config=private_endpoint_config,
        private_endpoint_auto_approval=pe_auto_approval,
        user_assigned_identity_for_cmk_encryption=user_assigned_identity_for_cmk_encryption,
        system_datastores_auth_mode=system_datastores_auth_mode,
        v1_legacy_mode=v1_legacy_mode)

    # TODO: Need to add a message that workspace created successfully.
    return workspace_object._get_create_status_dict()


def list_workspace(resource_group_name=None):

    from azureml._base_sdk_common.cli_wrapper._common import get_cli_specific_auth, get_default_subscription_id, \
        get_resource_group_or_default_name

    auth = get_cli_specific_auth()
    default_subscription_id = get_default_subscription_id(auth)

    # resource group can be None, as we create on user's behalf.
    resource_group_name = get_resource_group_or_default_name(
        resource_group_name, auth=auth)

    workspaces_dict = Workspace.list(default_subscription_id, auth=auth,
                                     resource_group=resource_group_name)
    serialized_workspace_list = list()
    for workspace_name in workspaces_dict:
        for workspace_object in workspaces_dict[workspace_name]:
            serialized_workspace_list.append(workspace_object._to_dict())

    return serialized_workspace_list


NO_WAIT = argument.Argument(
    "no_wait",
    "--no-wait", "",
    action="store_true",
    help="Do not wait for the workspace deletion to complete.")


DELETE_DEPENDENT_RESOURCES = argument.Argument(
    "delete_dependent_resources",
    "--all-resources", "",
    action="store_true",
    help="Deletes resources which this workspace depends on like storage, acr, kv and app insights.")

ACR_RESOURCE = argument.Argument(
    "container_registry",
    "--acr", "",
    help="The arm id of the container registry that you want to update this workspace with.")

FORCE_RESOURCE_UPDATE = argument.Argument(
    "force",
    "--force", "",
    action="store_true",
    help="Force update dependent resources without user's confirmation.")


@command(
    subgroup_type=WorkspaceSubGroup,
    command="delete",
    short_description="Delete a workspace.",
    argument_list=[
        DELETE_DEPENDENT_RESOURCES,
        NO_WAIT
    ])
def delete_workspace(
        workspace=None,
        delete_dependent_resources=False,
        no_wait=False,
        logger=None):

    return workspace.delete(
        delete_dependent_resources=delete_dependent_resources,
        no_wait=no_wait)


@command(
    subgroup_type=WorkspaceSubGroup,
    command="sync-keys",
    short_description="Sync workspace keys for dependent resources such as storage, acr, and app insights.")
def sync_workspace_keys(
        workspace=None,
        no_wait=False,
        # We should enforce a logger
        logger=None):

    return workspace._sync_keys(no_wait=no_wait)


@command(
    subgroup_type=WorkspaceSubGroup,
    command="diagnose",
    short_description="Diagnose workspace setup problems")
def diagnose_workspace(
        workspace=None,
        # We should enforce a logger
        logger=None):

    return workspace.diagnose_workspace(
        DiagnoseWorkspaceParameters(
            value=DiagnoseRequestProperties()))


@command(
    subgroup_type=WorkspaceSubGroup,
    command="list-keys",
    short_description="List workspace keys for dependent resources such as storage, acr, and app insights.")
def list_workspace_keys(
        workspace=None,
        # We should enforce a logger
        logger=None):

    return workspace.list_keys()


@command(
    subgroup_type=WorkspaceSubGroup,
    command="update-dependencies",
    short_description="Update workspace dependent resources.",
    argument_list=[
        ACR_RESOURCE,
        FORCE_RESOURCE_UPDATE
    ])
def update_workspace_dependencies(
        workspace=None,
        container_registry=None,
        force=False,
        # We should enforce a logger
        logger=None):
    return workspace.update_dependencies(
        container_registry=container_registry, force=force)


@command(
    subgroup_type=WorkspaceSubGroup,
    command="show",
    short_description="Show a workspace.")
def show_workspace(
        workspace=None,
        logger=None):

    return workspace.get_details()


USER = argument.Argument(
    "user",
    "--user",
    "",
    help="User with whom to share this workspace.",
    required=True)
ROLE = argument.Argument(
    "role",
    "--role",
    "",
    help="Role to assign to this user.",
    required=True)


@command(
    subgroup_type=WorkspaceSubGroup,
    command="share",
    short_description="Share a workspace with another user with a given role.",
    argument_list=[
        USER,
        ROLE
    ])
def share_workspace(
        workspace=None,
        user=None,
        role=None,
        logger=None):

    return workspace._share(user, role)


DESCRIPTION = argument.Argument(
    "description",
    "--description",
    "-d",
    help="Description of this workspace.",
    required=False)

TAGS = argument.Argument(
    "tags",
    "--tags",
    "",
    help="Tags associated with this workspace with 'key=value' syntax.",
    required=False)

IMAGE_BUILD_COMPUTE = argument.Argument(
    "image_build_compute",
    "--image-build-compute",
    "",
    help="Compute target for image build",
    required=False)

PRIVATE_USER_ASSIGNED_IDENTITY = argument.Argument(
    "primary_user_assigned_identity",
    "--primary-user-assigned-identity",
    "",
    help="The resourceId of the user assigned identity that used to represent workspace identity.",
    required=False)

ALLOW_PUBLIC_ACCESS_WHEN_BEHIND_VNET = argument.Argument(
    "allow_public_access_when_behind_vnet",
    "--allow-public-access",
    "",
    help="Allow public access to private link workspace",
    required=False)

V1_LEGACY_MODE = argument.Argument(
    "v1_legacy_mode",
    "--v1-legacy-mode",
    "-v",
    help="Prevent using v2 API service on public Azure Resource Manager if you set this parameter true.\
    Learn more at aka.ms/amlv2network",
    required=False)


@command(
    subgroup_type=WorkspaceSubGroup,
    command="update",
    short_description="Update a workspace.",
    argument_list=[
        argument.FRIENDLY_NAME,
        DESCRIPTION,
        TAGS,
        IMAGE_BUILD_COMPUTE,
        PRIVATE_USER_ASSIGNED_IDENTITY,
        ALLOW_PUBLIC_ACCESS_WHEN_BEHIND_VNET,
        V1_LEGACY_MODE
    ])
def update_workspace(
        workspace=None,
        friendly_name=None,
        description=None,
        tags=None,
        image_build_compute=None,
        primary_user_assigned_identity=None,
        allow_public_access_when_behind_vnet=None,
        v1_legacy_mode=None,
        logger=None):

    # Returns a dict containing the update details.
    return workspace.update(
        friendly_name=friendly_name,
        description=description,
        tags=get_tags_dict(tags),
        image_build_compute=image_build_compute,
        primary_user_assigned_identity=primary_user_assigned_identity,
        allow_public_access_when_behind_vnet=allow_public_access_when_behind_vnet,
        v1_legacy_mode=v1_legacy_mode)


def get_tags_dict(tags_str):
    tags_dict = dict()
    if tags_str:
        tags = tags_str.split()
        for tag in tags:
            key, value = tag.split("=", 1)
            tags_dict[key] = value

    return tags_dict
