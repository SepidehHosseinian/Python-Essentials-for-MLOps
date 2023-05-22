# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli.workspace.private_endpoint.private_endpoint_subgroup import WorkspacePrivateEndPointSubGroup
from azureml._cli.cli_command import command
from azureml._cli.cli_command import argument
from azureml._cli.workspace.private_endpoint.private_endpoint_arguments import PE_NAME_ARGUMENT, \
    PE_LOCATION_ARGUMENT, PE_VNET_NAME_ARGUMENT, PE_SUBNET_NAME_ARGUMENT, \
    PE_SUBSCRIPTION_ID_ARGUMENT, PE_RESOURCE_GROUP_ARGUMENT, PE_AUTO_APPROVAL_ARGUMENT, PE_CONNECTION_NAME, TAGS


from azureml.core.private_endpoint import PrivateEndPointConfig
from azureml.core.workspace import Workspace


@command(
    subgroup_type=WorkspacePrivateEndPointSubGroup,
    command="add",
    short_description="Add private endpoint to a workspace.",
    argument_list=[
        PE_NAME_ARGUMENT,
        PE_LOCATION_ARGUMENT,
        PE_VNET_NAME_ARGUMENT,
        PE_SUBNET_NAME_ARGUMENT,
        PE_SUBSCRIPTION_ID_ARGUMENT,
        PE_RESOURCE_GROUP_ARGUMENT,
        PE_AUTO_APPROVAL_ARGUMENT,
        TAGS
    ])
def workspace_add_private_endpoint(
        workspace=None,
        pe_name=None,
        pe_location=None,
        pe_vnet_name=None,
        pe_subnet_name=None,
        pe_subscription_id=None,
        pe_resource_group=None,
        pe_auto_approval=None,
        logger=None,
        tags=None):

    if pe_name is not None:
        private_endpoint_config = PrivateEndPointConfig(
            pe_name, pe_vnet_name, pe_subnet_name, pe_subscription_id, pe_resource_group)
        workspace.add_private_endpoint(
            private_endpoint_config=private_endpoint_config,
            private_endpoint_auto_approval=pe_auto_approval,
            location=pe_location,
            tags=get_tags_dict(tags))


@command(
    subgroup_type=WorkspacePrivateEndPointSubGroup,
    command="list",
    short_description="List all private endpoints in a workspace.",
    argument_list=[
    ])
def workspace_list_private_endpoint(
        workspace=None,
        logger=None):

    private_endpoints = workspace.private_endpoints
    serialized_pe_list = list()
    for pe in private_endpoints:
        serialized_pe_list.append(private_endpoints[pe].__dict__)

    return serialized_pe_list


@command(
    subgroup_type=WorkspacePrivateEndPointSubGroup,
    command="delete",
    short_description="Delete the specified private endpoint Connection in the workspace.",
    argument_list=[
        argument.SUBSCRIPTION_ID,
        argument.RESOURCE_GROUP_NAME,
        argument.WORKSPACE_NAME,
        PE_CONNECTION_NAME])
def workspace_delete_private_endpoint(
        subscription_id,
        resource_group_name,
        workspace_name,
        pe_connection_name,
        logger=None):
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group_name,
        workspace_name=workspace_name)
    workspace.delete_private_endpoint_connection(pe_connection_name)


def get_tags_dict(tags_str):
    tags_dict = dict()
    if tags_str:
        tags = tags_str.split()
        for tag in tags:
            key, value = tag.split("=", 1)
            tags_dict[key] = value

    return tags_dict
