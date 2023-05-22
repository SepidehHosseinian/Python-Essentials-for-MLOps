# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the base functionality for datastores that save connection information to Azure SQL database."""

from .abstract_datastore import AbstractDatastore


class AzureSqlDatabaseDatastore(AbstractDatastore):
    """Represents a datastore that saves connection information to Azure SQL Database.

    You should not work with this class directly. To create a datastore that saves connection information to
    Azure SQL Database, use the :meth:`azureml.core.datastore.Datastore.register_azure_sql_database` method
    of the :class:`azureml.core.datastore.Datastore` class.

    Note: When using a datastore to access data, you must have permission to access the data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param server_name: The SQL server name
    :type server_name: str
    :param database_name: The SQL database name
    :type database_name: str
    :param tenant_id: The Directory ID/Tenant ID of the service principal.
    :type tenant_id: str
    :param client_id: The Client ID/Application ID of the service principal.
    :type client_id: str
    :param client_secret: The secret of the service principal.
    :type client_secret: str
    :param resource_url: The resource URL, which determines what operations will be performed on
        the SQL database store. If None, defaults to https://database.windows.net/.
    :type resource_url: str, optional
    :param authority_url: The authority URL used to authenticate the user. Defaults to
        https://login.microsoftonline.com.
    :type authority_url: str, optional
    :param username: The username of the database user to access the database.
    :type username: str
    :param password: The password of the database user to access the database.
    :type password: str
    """

    def __init__(self, workspace, name, server_name, database_name,
                 tenant_id=None, client_id=None, client_secret=None,
                 resource_url=None, authority_url=None, username=None, password=None, auth_type=None,
                 service_data_access_auth_identity=None):
        """Initialize a new Azure SQL Database Datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param server_name: The SQL server name
        :type server_name: str
        :param database_name: The SQL database name
        :type database_name: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal.
        :type tenant_id: str
        :param client_id: The Client ID/Application ID of the service principal.
        :type client_id: str
        :param client_secret: The secret of the service principal.
        :type client_secret: str
        :param resource_url: The resource URL, which determines what operations will be performed on
            the SQL database store. If None, defaults to https://database.windows.net/.
        :type resource_url: str, optional
        :param authority_url: The authority URL used to authenticate the user. Defaults to
            https://login.microsoftonline.com.
        :type authority_url: str, optional
        :param username: The username of the database user to access the database.
        :type username: str
        :param password: The password of the database user to access the database.
        :type password: str
        :param auth_type: The authentication type.
        :type auth_type: str
        :param service_data_access_auth_identity: Indicates which identity to use
            to authenticate service data access to customer's storage. Possible values
            include: 'None', 'WorkspaceSystemAssignedIdentity', 'WorkspaceUserAssignedIdentity'
        :type service_data_access_auth_identity: str or
            ~_restclient.models.ServiceDataAccessAuthIdentity
        """
        import azureml.data.constants as constants

        super(AzureSqlDatabaseDatastore, self).__init__(workspace, name, constants.AZURE_SQL_DATABASE)
        self.server_name = server_name
        self.database_name = database_name
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource_url = resource_url
        self.authority_url = authority_url
        self.username = username
        self.password = password
        self.auth_type = auth_type
        self.service_data_access_auth_identity = service_data_access_auth_identity

    def _as_dict(self, hide_secret=True):
        output = super(AzureSqlDatabaseDatastore, self)._as_dict()
        output["server_name"] = self.server_name
        output["database_name"] = self.database_name
        output["resource_url"] = self.resource_url

        if self.auth_type == "SqlAuthentication":
            output["username"] = self.username
            if not hide_secret:
                output["password"] = self.password
        elif self.auth_type == "ServicePrincipal":
            output["tenant_id"] = self.tenant_id
            output["client_id"] = self.client_id
            output["authority_url"] = self.authority_url
            if not hide_secret:
                output["client_secret"] = self.client_secret

        if self.service_data_access_auth_identity:
            output["service_data_access_auth_identity"] = self.service_data_access_auth_identity
        return output
