# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the base functionality for datastores that save connection information to Azure Data Lake Storage."""

from .abstract_datastore import AbstractDatastore


class AbstractADLSDatastore(AbstractDatastore):
    """Represents the base class for datastores that save connection information to Azure Data Lake Storage.

    You should not work with this class directly. To create a datastore that saves connection information
    to Azure Data Lake Storage, use one of the ``register_azure_data_lake*`` methods of the
    :class:`azureml.core.datastore.Datastore` class.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param datastore_type: The datastore type, for example, "AzureDataLake" or "AzureDataLakeGen2".
    :type datastore_type: str
    :param tenant_id: The Directory ID/Tenant ID of the service principal.
    :type tenant_id: str
    :param client_id: The Client ID/Application ID of the service principal.
    :type client_id: str
    :param client_secret: The secret of the service principal.
    :type client_secret: str
    :param resource_url: The resource url, which determines what operations will be performed on
        the Data Lake Store.
    :type resource_url: str
    :param authority_url: The authorization server's url, defaults to https://login.microsoftonline.com.
    :type authority_url: str
    """

    def __init__(self, workspace, name, datastore_type, tenant_id, client_id, client_secret, resource_url,
                 authority_url, service_data_access_auth_identity):
        """Initialize a new Azure Data Lake Datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param datastore_type: The datastore type, for example, "AzureDataLake" or "AzureDataLakeGen2".
        :type datastore_type: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal.
        :type tenant_id: str
        :param client_id: The Client ID/Application ID of the service principal.
        :type client_id: str
        :param client_secret: The secret of the service principal.
        :type client_secret: str
        :param resource_url: The resource URL, which determines what operations will be performed on
            the Data Lake Store.
        :type resource_url: str
        :param authority_url: The authorization server's url, defaults to https://login.microsoftonline.com.
        :type authority_url: str
        :param service_data_access_auth_identity: Indicates which identity to use
            to authenticate service data access to customer's storage. Possible values
            include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
        :type service_data_access_auth_identity: str or
            ~_restclient.models.ServiceDataAccessAuthIdentity
        """
        super(AbstractADLSDatastore, self).__init__(workspace, name, datastore_type)
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource_url = resource_url
        self.authority_url = authority_url
        self.service_data_access_auth_identity = service_data_access_auth_identity

    def _as_dict(self, hide_secret=True):
        output = super(AbstractADLSDatastore, self)._as_dict()
        output["tenant_id"] = self.tenant_id
        output["client_id"] = self.client_id
        output["resource_url"] = self.resource_url
        output["authority_url"] = self.authority_url

        if self.service_data_access_auth_identity:
            output["service_data_access_auth_identity"] = self.service_data_access_auth_identity

        if not hide_secret:
            output["client_secret"] = self.client_secret

        return output


class AzureDataLakeDatastore(AbstractADLSDatastore):
    """Represents a datastore that saves connection information to Azure Data Lake Storage.

    To create a datastore that saves connection information to Azure Data Lake Storage, use the
    :meth:`azureml.core.datastore.Datastore.register_azure_data_lake` method
    of the :class:`azureml.core.datastore.Datastore` class.

    Note: When using a datastore to access data, you must have permission to access the data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param store_name: The Azure Data Lake store name.
    :type store_name: str
    :param tenant_id: The Directory ID/Tenant ID of the service principal.
    :type tenant_id: str
    :param client_id: The Client ID/Application ID of the service principal.
    :type client_id: str
    :param client_secret: The secret of the service principal.
    :type client_secret: str
    :param resource_url: The resource url, which determines what operations will be performed on
        the Data Lake Store.
    :type resource_url: str, optional
    :param authority_url: The authority URL used to authenticate the user.
    :type authority_url: str, optional
    :param subscription_id: The ID of the subscription the ADLS store belongs to.
        Specify both ``subscription_id`` and ``resource_group`` when using the AzureDataLakeDatastore
        as a data transfer destination.
    :type subscription_id: str, optional
    :param resource_group: The resource group the ADLS store belongs to.
        Specify both ``subscription_id`` and ``resource_group`` when using the AzureDataLakeDatastore
        as a data transfer destination.
    :type resource_group: str, optional
    :param service_data_access_auth_identity: Indicates which identity to use
        to authenticate service data access to customer's storage. Possible values
        include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
    :type service_data_access_auth_identity: str or
        ~_restclient.models.ServiceDataAccessAuthIdentity
    """

    def __init__(self, workspace, name, store_name, tenant_id, client_id, client_secret,
                 resource_url=None, authority_url=None, subscription_id=None, resource_group=None,
                 service_data_access_auth_identity=None):
        """Initialize a new Azure Data Lake Datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param store_name: The Azure Data Lake store name.
        :type store_name: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal.
        :type tenant_id: str
        :param client_id: The Client ID/Application ID of the service principal.
        :type client_id: str
        :param client_secret: The secret of the service principal.
        :type client_secret: str
        :param resource_url: The resource url, which determines what operations will be performed on
            the Data Lake Store.
        :type resource_url: str, optional
        :param authority_url: The authority URL used to authenticate the user.
        :type authority_url: str, optional
        :param subscription_id: The ID of the subscription the ADLS store belongs to.
            Specify both ``subscription_id`` and ``resource_group`` when using the AzureDataLakeDatastore
            as a data transfer destination.
        :type subscription_id: str, optional
        :param resource_group: The resource group the ADLS store belongs to.
            Specify both ``subscription_id`` and ``resource_group`` when using the AzureDataLakeDatastore
            as a data transfer destination.
        :type resource_group: str, optional
        :param service_data_access_auth_identity: Indicates which identity to use
            to authenticate service data access to customer's storage. Possible values
            include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
        :type service_data_access_auth_identity: str or
            ~_restclient.models.ServiceDataAccessAuthIdentity
        """
        import azureml.data.constants as constants

        super(AzureDataLakeDatastore, self).__init__(workspace, name, constants.AZURE_DATA_LAKE, tenant_id, client_id,
                                                     client_secret, resource_url, authority_url,
                                                     service_data_access_auth_identity)
        self.store_name = store_name
        self._subscription_id = subscription_id
        self._resource_group = resource_group

    @property
    def subscription_id(self):
        """Return the subscription ID of the ADLS store.

        :return: The subscription ID of the ADLS store.
        :rtype: str
        """
        return self._subscription_id

    @subscription_id.setter
    def subscription_id(self, subscription_id):
        """Obsolete, will be removed in future releases."""
        self._subscription_id = subscription_id

    @property
    def resource_group(self):
        """Return the resource group of the ADLS store.

        :return: The resource group of the ADLS store
        :rtype: str
        """
        return self._resource_group

    @resource_group.setter
    def resource_group(self, resource_group):
        """Obsolete, will be removed in future releases."""
        self._resource_group = resource_group

    def _as_dict(self, hide_secret=True):
        output = super(AzureDataLakeDatastore, self)._as_dict(hide_secret)
        output["store_name"] = self.store_name
        output["subscription_id"] = self.subscription_id
        output["resource_group"] = self.resource_group
        return output


class AzureDataLakeGen2Datastore(AbstractADLSDatastore):
    """Represents a datastore that saves connection information to Azure Data Lake Storage Gen2.

    To create a datastore that saves connection information to Azure Data Lake Storage, use the
    ``register_azure_data_lake_gen2`` method of the :class:`azureml.core.datastore.Datastore` class.

    To access data from an AzureDataLakeGen2Datastore object, create a :class:`azureml.core.Dataset` and use
    one of the methods like :meth:`azureml.data.dataset_factory.FileDatasetFactory.from_files` for a FileDataset.
    For more information, see `Create Azure Machine Learning datasets
    <https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets>`_.

    Also keep in mind:

    * The AzureDataLakeGen2 class does not provide upload method, recommended way to uploading data to
      AzureDataLakeGen2 datastores is via Dataset upload. More details could be found at :
      https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets

    * When using a datastore to access data, you must have permission to access the data, which depends on the
      credentials registered with the datastore.

    * When using Service Principal Authentication to access storage via AzureDataLakeGen2, the service principal
      or app registration must be assigned the specific role-based access control (RBAC) role at minimum of
      "Storage Blob Data Reader". For more information, see `Storage built-in roles
      <https://docs.microsoft.com/azure/role-based-access-control/built-in-roles#storage>`_.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param container_name: The name of the Azure blob container.
    :type container_name: str
    :param account_name: The storage account name.
    :type account_name: str
    :param tenant_id: The Directory ID/Tenant ID of the service principal.
    :type tenant_id: str
    :param client_id: The Client ID/Application ID of the service principal.
    :type client_id: str
    :param client_secret: The secret of the service principal.
    :type client_secret: str
    :param resource_url: The resource url, which determines what operations will be performed on
        the Data Lake Store.
    :type resource_url: str
    :param authority_url: The authority URL used to authenticate the user.
    :type authority_url: str
    :param protocol: The protocol to use to connect to the blob container.
        If None, defaults to https.
    :type protocol: str
    :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
    :type endpoint: str
    :param service_data_access_auth_identity: Indicates which identity to use
            to authenticate service data access to customer's storage. Possible values
            include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
    :type service_data_access_auth_identity: str or
            ~_restclient.models.ServiceDataAccessAuthIdentity
    """

    def __init__(self, workspace, name, container_name, account_name, tenant_id=None, client_id=None,
                 client_secret=None, resource_url=None, authority_url=None, protocol=None, endpoint=None,
                 service_data_access_auth_identity=None):
        """Initialize a new Azure Data Lake Gen2 Datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param container_name: The name of the Azure blob container.
        :type container_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal.
        :type tenant_id: str
        :param client_id: The Client ID/Application ID of the service principal.
        :type client_id: str
        :param client_secret: The secret of the service principal.
        :type client_secret: str
        :param resource_url: The resource url, which determines what operations will be performed on
            the Data Lake Store.
        :type resource_url: str
        :param authority_url: The authority URL used to authenticate the user.
        :type authority_url: str
        :param protocol: The protocol to use to connect to the blob container.
            If None, defaults to https.
        :type protocol: str
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
        :type endpoint: str
        :param service_data_access_auth_identity: Indicates which identity to use
            to authenticate service data access to customer's storage. Possible values
            include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
        :type service_data_access_auth_identity: str or
            ~_restclient.models.ServiceDataAccessAuthIdentity
        """
        import azureml.data.constants as constants

        super(AzureDataLakeGen2Datastore, self).__init__(workspace, name, constants.AZURE_DATA_LAKE_GEN2, tenant_id,
                                                         client_id, client_secret, resource_url, authority_url,
                                                         service_data_access_auth_identity)
        self.container_name = container_name
        self.account_name = account_name
        self.protocol = protocol
        self.endpoint = endpoint

    def _as_dict(self, hide_secret=True):
        output = super(AzureDataLakeGen2Datastore, self)._as_dict(hide_secret)
        output["container_name"] = self.container_name
        output["account_name"] = self.account_name
        output["protocol"] = self.protocol
        output["endpoint"] = self.endpoint
        return output
