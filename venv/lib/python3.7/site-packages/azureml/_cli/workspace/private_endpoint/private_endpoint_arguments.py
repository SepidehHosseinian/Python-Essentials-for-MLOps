# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli import argument


PE_NAME_ARGUMENT = argument.Argument(
    "pe-name",
    "--pe-name",
    "",
    help="Name of the workspace private endpoint. "
         "Use this parameter to restrict workspace access to private networks, via a private endpoint.")
PE_LOCATION_ARGUMENT = argument.Argument(
    "pe-location",
    "--pe-location",
    "",
    help="Location of the workspace private endpoint. "
         "If not specified it will be the same location of the workspace.")
PE_VNET_NAME_ARGUMENT = argument.Argument(
    "pe-vnet-name",
    "--pe-vnet-name",
    "",
    help="Name of the existing vnet to create the workspace private endpoint in. ")
PE_SUBNET_NAME_ARGUMENT = argument.Argument(
    "pe-subnet-name",
    "--pe-subnet-name",
    "",
    default="default",
    help="Name of the existing subnet to create the workspace private endpoint in. ")
PE_SUBSCRIPTION_ID_ARGUMENT = argument.Argument(
    "pe-subscription-id",
    "--pe-subscription-id",
    "",
    help="Id of the existing subscription to create the workspace private endpoint in. "
         "The vnet should be in the same subscription. If not specified, the subscription Id of the workspace "
         "will be used.")
PE_RESOURCE_GROUP_ARGUMENT = argument.Argument(
    "pe-resource-group",
    "--pe-resource-group",
    "",
    help="Name of the existing resource group to create the workspace private endpoint in. "
         "The vnet should be in the same resource group. If not specified, the resource group of the workspace "
         "will be used.")
PE_AUTO_APPROVAL_ARGUMENT = argument.Argument(
    "pe-auto-approval",
    "--pe-auto-approval",
    "",
    action="store_true",
    help="Whether private endpoint connections to the workspace resource via a private link should be "
         "auto approved.")
PE_CONNECTION_RESOURCE_ID = argument.Argument(
    "resource-id",
    "--resource-id",
    "",
    help="The resource id of the Private EndPoint Connection.")
PE_CONNECTION_NAME = argument.Argument(
    "pe_connection_name",
    "--pe-connection-name",
    "",
    help="The name of the Private EndPoint Connection.")

TAGS = argument.Argument(
    "tags",
    "--tags",
    "",
    help="Tags associated with this private endpoint with 'key=value' syntax.")
