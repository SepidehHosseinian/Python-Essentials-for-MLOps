# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the base functionality for datastores that save connection information to Azure Database for MySQL."""

from .abstract_datastore import AbstractDatastore


class AzureMySqlDatastore(AbstractDatastore):
    """Represents a datastore that saves connection information to Azure Database for MySQL.

    You should not use this class directly. To create a datastore that saves connection information to Azure Database
    for MySql, use the :meth:`azureml.core.datastore.Datastore.register_azure_my_sql` method
    of the :class:`azureml.core.datastore.Datastore` class.

    Note: When using a datastore to access data, you must have permission to access the data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param server_name: The MySQL server name.
    :type server_name: str
    :param database_name: The MySQL database name.
    :type database_name: str
    :param user_id: The user ID of the MySQL server.
    :type user_id: str
    :param user_password: The user password of the MySQL server.
    :type user_password: str
    :param port_number: The port number of the MySQL server.
    :type port_number: str
    """

    def __init__(self, workspace, name, server_name, database_name,
                 user_id, user_password, port_number=None):
        """Initialize a new AzureMySqlDatastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param server_name: The MySQL server name.
        :type server_name: str
        :param database_name: The MySQL database name.
        :type database_name: str
        :param user_id: The user ID of the MySQL server.
        :type user_id: str
        :param user_password: The user password of the MySQL server.
        :type user_password: str
        :param port_number: The port number of the MySQL server.
        :type port_number: str
        """
        import azureml.data.constants as constants

        super(AzureMySqlDatastore, self).__init__(workspace, name, constants.AZURE_MYSQL)
        self.server_name = server_name
        self.database_name = database_name
        self.user_id = user_id
        self.user_password = user_password
        self.port_number = port_number

    def _as_dict(self, hide_secret=True):
        output = super(AzureMySqlDatastore, self)._as_dict()
        output["server_name"] = self.server_name
        output["database_name"] = self.database_name
        output["user_id"] = self.user_id
        output["port_number"] = self.port_number

        if not hide_secret:
            output["user_password"] = self.user_password

        return output
