# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for defining and configuring Azure Private EndPoints."""

import re
import logging
from azureml._project import _commands
from azureml.core.authentication import InteractiveLoginAuthentication

module_logger = logging.getLogger(__name__)


class PrivateEndPointConfig(object):
    """Defines configuration for an Azure Private EndPoint.

    `Azure Private Endpoint <https://docs.microsoft.com/azure/private-link/private-endpoint-overview>`_ is a network
    interface that connects you privately and securely to a Azure ML workspace with Private Link.

    :param name: The name of the PrivateEndPoint.
    :type name: str
    :param vnet_name: The name of the VNet.
    :type vnet_name: str
    :param vnet_subnet_name: The name of the subnet to deploy and allocate private IP addresses from.
    :type vnet_subnet_name: str
    :param vnet_subscription_id: The Azure subscription id containing the VNet. If not specified,
        the subscription ID of workspace will be taken.
    :type vnet_subscription_id: str
    :param vnet_resource_group: The resource group containing the VNet. If not specified,
        the resource group of workspace will be taken.
    :type vnet_resource_group: str
    """

    def __init__(
            self,
            name,
            vnet_name,
            vnet_subnet_name=None,
            vnet_subscription_id=None,
            vnet_resource_group=None):
        """
        Initialize PrivateEndPointConfig.

        :param name: The name of the PrivateEndPoint.
        :type name: str
        :param vnet_name: The name of the VNet.
        :type vnet_name: str
        :param vnet_subnet_name: The name of the subnet to deploy and allocate private IP addresses from.
        :type vnet_subnet_name: str
        :param vnet_subscription_id: The Azure subscription id containing the VNet. If not specified,
            the subscription ID of workspace will be taken.
        :type vnet_subscription_id: str
        :param vnet_resource_group: The resource group containing the VNet. If not specified,
            the resource group of workspace will be taken.
        :type vnet_resource_group: str
        """
        self.name = name
        self.vnet_name = vnet_name
        self.vnet_subnet_name = vnet_subnet_name if vnet_subnet_name else "default"
        # Once cross resource group issue is fixed, these two params will be
        # used.
        self.vnet_subscription_id = vnet_subscription_id
        self.vnet_resource_group = vnet_resource_group


class PrivateEndPoint(object):
    """Defines a private endpoint for managing private endpoint connections associated with an Azure ML workspace.

    :param private_endpoint_connection_resource_id: The ARM resource ID of the private endpoint connection.
    :type private_endpoint_connection_resource_id: str
    :param private_endpoint_resource_id: The ARM resource ID of the private endpoint.
    :type private_endpoint_resource_id: str
    """

    def __init__(
            self,
            private_endpoint_connection_resource_id=None,
            private_endpoint_resource_id=None):
        """
        Initialize PrivateEndPoint.

        :param private_endpoint_connection_resource_id: The ARM resource ID of the private endpoint connection.
        :type private_endpoint_connection_resource_id: str
        :param private_endpoint_resource_id: The ARM resource ID of the private endpoint.
        :type private_endpoint_resource_id: str

        """
        self.private_endpoint_connection_resource_id = private_endpoint_connection_resource_id
        self.private_endpoint_resource_id = private_endpoint_resource_id

    def get_details(self, auth=None):
        """Get details of the private end point.

        :param auth: The authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication
            or azureml.core.authentication.InteractiveLoginAuthentication

        :return: Private endpoint details in dictionary format.
        :rtype: dict[str, str]
        """
        if not auth:
            auth = InteractiveLoginAuthentication()
        return _commands.get_private_endpoint_details(
            auth, self.private_endpoint_resource_id)

    def delete(self, auth=None):
        """Delete the private endpoint connection to the workspace.

        .. note::

            This function is deprecated and please use Workspace.delete_private_endpoint_connection


        :param auth: The authentication object. For more details refer to https://aka.ms/aml-notebook-auth.
            If None, the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication
            or azureml.core.authentication.InteractiveLoginAuthentication

        :return: None if successful; otherwise, throws an error.
        :rtype: None
        """
        module_logger.warning(
            "'PrivateEndPoint.delete' is deprecated and will be removed in a future release. "
            "Use Workspace.delete_private_endpoint_connection to delete private endpoint connection.")
        if not auth:
            auth = InteractiveLoginAuthentication()
        scope = re.search(
            r'/subscriptions/(.+)/resourceGroups/(.+)/providers'
            + r'/Microsoft\.MachineLearningServices/workspaces/(.+)/privateEndpointConnections/(.+)',
            self.private_endpoint_connection_resource_id,
            re.IGNORECASE)
        if scope:
            subscription_id = scope.group(1)
            resource_group = scope.group(2)
            workspace_name = scope.group(3)
            pe_connection_name = scope.group(4)

        _commands.delete_workspace_private_endpoint_connection(
            auth, subscription_id, resource_group, workspace_name, pe_connection_name)
