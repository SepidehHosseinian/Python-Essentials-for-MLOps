# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for creating and managing linked service in AML workspace."""

from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml._base_sdk_common.workspace.operations import LinkedServicesOperations
from azureml._base_sdk_common.workspace import AzureMachineLearningWorkspaces
from azureml._base_sdk_common.workspace.models import LinkedServiceProps, LinkedServiceRequest, Identity,\
    ResourceIdentityType
from azureml._base_sdk_common.common import check_valid_resource_name
from azureml.exceptions import UserErrorException


@experimental
class LinkedService(object):
    """Defines an Resource for managing linking between AML workspace with other services on Azure."""

    def __init__(self,
                 workspace,
                 name,
                 type,
                 linked_service_resource_id,
                 system_assigned_identity_principal_id):
        """
        Initialize LinkedService object.

        :param workspace: The AzureML workspace in which the linked service exists.
        :type workspace: azureml.core.Workspace
        :param name: The name of the linked service.
        :type name: str
        :param type: type of the linked service.
        :type type: str
        :param linked_service_resource_id: arm resource id of the linked service.
        :type linked_service_resource_id: str
        :param system_assigned_identity_principal_id: system assigned identity principal id of the linked service.
        :type system_assigned_identity_principal_id: str
        """
        self._workspace = workspace
        self._name = name
        self._type = type
        self._linked_service_resource_id = linked_service_resource_id
        self._system_assigned_identity_principal_id = system_assigned_identity_principal_id

    @property
    def name(self):
        """Return the linked service name.

        :return: The linked service name.
        :rtype: str
        """
        return self._name

    @property
    def type(self):
        """Return the linked service type, such as Synapse.

        :return: The linked service type.
        :rtype: str
        """
        return self._type

    @property
    def linked_service_resource_id(self):
        """Return the linked service resource id.

        :return: The linked service resource id.
        :rtype: str
        """
        return self._linked_service_resource_id

    @property
    def system_assigned_identity_principal_id(self):
        """Return the linked service system assigned identity principal id.

        :return: The linked service system assigned identity principal id.
        :rtype: str
        """
        return self._system_assigned_identity_principal_id

    @staticmethod
    def register(workspace,
                 name,
                 linked_service_config):
        """Register a linked service under the given workspace.

        :param workspace: The AzureML workspace in which the linked service is to be registered.
        :type workspace: azureml.core.Workspace
        :param name: The name of the linked service.
        :type name: str
        :param linked_service_config: The configuration linked service.
        :type linked_service_config: azureml.core.linked_service.LinkedServiceConfiguration
        :return: A linked service under the given workspace.
        :rtype: azureml.core.LinkedService
        """
        check_valid_resource_name(name, "workspace")

        if not linked_service_config or not isinstance(linked_service_config, LinkedServiceConfiguration):
            raise UserErrorException(
                'A linked_service_config is required to add linked service to AML workspace.')

        if not isinstance(linked_service_config, SynapseWorkspaceLinkedServiceConfiguration):
            raise UserErrorException(
                'Currently only synapse workspace linked service is supported.')

        linked_service_resource_id = linked_service_config._get_resource_id()
        linked_service_pros = LinkedServiceProps(linked_service_resource_id)
        linked_service_request = LinkedServiceRequest(
            name=name,
            location=workspace.location,
            identity=Identity(type=ResourceIdentityType.system_assigned),
            properties=linked_service_pros)

        resp = LinkedServicesOperations.create(
            workspace._auth._get_service_client(
                AzureMachineLearningWorkspaces,
                workspace._subscription_id).linked_services,
            workspace._subscription_id,
            workspace._resource_group,
            workspace._workspace_name,
            name,
            linked_service_request)

        return LinkedService(
            workspace,
            resp.name,
            resp.properties.link_type,
            resp.properties.linked_service_resource_id,
            resp.identity.principal_id)

    @staticmethod
    def get(workspace, name):
        """Get the linked service under the given workspace based on link name.

        :param workspace: The AzureML workspace in which the linked service is to be fetched.
        :type workspace: azureml.core.Workspace
        :param name: A name of a linked service.
        :type name: str
        :return: A linked service under the given workspace.
        :rtype: azureml.core.LinkedService
        """
        resp = LinkedServicesOperations.get(
            workspace._auth._get_service_client(
                AzureMachineLearningWorkspaces,
                workspace._subscription_id).linked_services,
            workspace._subscription_id,
            workspace._resource_group,
            workspace._workspace_name,
            name)
        return LinkedService(
            workspace,
            resp.name,
            resp.properties.link_type,
            resp.properties.linked_service_resource_id,
            resp.identity.principal_id)

    @staticmethod
    def list(workspace):
        """List the linked services under the given workspace.

        :param workspace: The AzureML workspace in which the linked service is to be listed.
        :type workspace: azureml.core.Workspace
        :return: A list of the linked services under the given workspace.
        :rtype: [azureml.core.LinkedService]
        """
        return [LinkedService(
            workspace,
            resp.name,
            resp.properties.link_type,
            resp.properties.linked_service_resource_id,
            resp.identity.principal_id)
            for resp in LinkedServicesOperations.list(
                workspace._auth._get_service_client(
                    AzureMachineLearningWorkspaces,
                    workspace._subscription_id).linked_services,
                workspace._subscription_id,
                workspace._resource_group,
                workspace._workspace_name).value]

    def unregister(self):
        """Delink this linked service."""
        LinkedServicesOperations.delete(
            self._workspace._auth._get_service_client(
                AzureMachineLearningWorkspaces,
                self._workspace._subscription_id).linked_services,
            self._workspace._subscription_id,
            self._workspace._resource_group,
            self._workspace._workspace_name,
            self._name)

    def __repr__(self):
        """Produce an unambiguous 'formal' string representation of this linked service.

        :return: A string representation of the linked service.
        :rtype: str
        """
        return '{}(workspace={}, name={}, type={}, linked_service_resource_id={}, '\
            'system_assigned_identity_principal_id={}'.format(
                self.__class__.__name__,
                self._workspace.__repr__(),
                self._name,
                self._type,
                self._linked_service_resource_id,
                self._system_assigned_identity_principal_id)


@experimental
class LinkedServiceConfiguration(object):
    """Defines a parent class of linked service configuration."""

    pass


@experimental
class SynapseWorkspaceLinkedServiceConfiguration(LinkedServiceConfiguration):
    """Defines a linked service configuration for linking synapse workspace."""

    def __init__(self,
                 subscription_id,
                 resource_group,
                 name):
        """
        Initialize SynapseWorkspaceLinkedServiceConfiguration object.

        :param subscription_id: The subscription id of the synapse workspace.
        :type subscription_id: str
        :param resource_group: The resource group of the synapse workspace.
        :type resource_group: str
        :param name: The name of the synapse workspace.
        :type name: str
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.name = name

    def _get_resource_id(self):
        SYNAPSE_RESOURCE_ID_FMT = (
            '/subscriptions/{}/resourcegroups/{}/providers/Microsoft.Synapse/workspaces/{}')
        return SYNAPSE_RESOURCE_ID_FMT.format(self.subscription_id, self.resource_group, self.name)
