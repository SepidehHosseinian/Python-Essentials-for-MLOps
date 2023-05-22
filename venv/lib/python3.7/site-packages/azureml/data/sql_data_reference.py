# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for creating references to data in datastores that save connection info to SQL databases."""

from azureml.data.data_reference import DataReference


class SqlDataReference(DataReference):
    """Represents a reference to data in a datastore that saves connection information to a SQL database.

    :param datastore: The Datastore to reference.
    :type datastore: azureml.data.azure_sql_database_datastore.AzureSqlDatabaseDatastore
                or azureml.data.azure_postgre_sql_datastore.AzurePostgreSqlDatastore
                or azureml.data.azure_my_sql_datastore.AzureMySqlDatastore
    :param data_reference_name: The name of the data reference.
    :type data_reference_name: str
    :param sql_table: The name of the table in SQL database.
    :type sql_table: str, optional
    :param sql_query: The SQL query to use with the SQL database.
    :type sql_query: str, optional
    :param sql_stored_procedure: The name of the stored procedure invoke in the SQL database.
    :type sql_stored_procedure: str, optional
    :param sql_stored_procedure_params: An optional list of parameters to pass to the stored procedure.
    :type sql_stored_procedure_params: [azureml.data.stored_procedure_parameter.StoredProcedureParameter], optional
    """

    def __init__(self, datastore, data_reference_name=None,
                 sql_table=None, sql_query=None, sql_stored_procedure=None, sql_stored_procedure_params=None):
        """Initialize sql data reference.

        :param datastore: The Datastore to reference.
        :type datastore: azureml.data.azure_sql_database_datastore.AzureSqlDatabaseDatastore
                    or azureml.data.azure_postgre_sql_datastore.AzurePostgreSqlDatastore
                    or azureml.data.azure_my_sql_datastore.AzureMySqlDatastore
        :param data_reference_name: The name of the data reference.
        :type data_reference_name: str
        :param sql_table: The name of the table in SQL database.
        :type sql_table: str, optional
        :param sql_query: The SQL query to use with the SQL database.
        :type sql_query: str, optional
        :param sql_stored_procedure: The name of the stored procedure invoke in the SQL database.
        :type sql_stored_procedure: str, optional
        :param sql_stored_procedure_params: An optional list of parameters to pass to stored procedure.
        :type sql_stored_procedure_params: [azureml.data.stored_procedure_parameter.StoredProcedureParameter], optional
        """
        self.sql_table = sql_table
        self.sql_query = sql_query
        self.sql_stored_procedure = sql_stored_procedure
        self.sql_stored_procedure_params = sql_stored_procedure_params

        super(SqlDataReference, self).__init__(datastore, data_reference_name=data_reference_name)

    def path(self, path=None, data_reference_name=None):
        """Create a SqlDataReference instance based on the given path. Not supported for SqlDataReference.

        :param path: The path on the datastore.
        :type path: str
        :param data_reference_name: The name of the data reference.
        :type data_reference_name: str
        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        raise NotImplementedError("SqlDataReference does not support `path` operation.")

    def as_download(self, path_on_compute=None, overwrite=False):
        """Switch data reference operation to download. Not supported for SqlDataReference.

        :param path_on_compute: The path on the compute for the data reference.
        :type path_on_compute: str
        :param overwrite: Indicates whether to overwrite existing data.
        :type overwrite: bool
        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        raise NotImplementedError("SqlDataReference does not support `download` operation.")

    def as_upload(self, path_on_compute=None, overwrite=False):
        """Switch data reference operation to upload. Not supported for SqlDataReference.

        :param path_on_compute: The path on the compute for the data reference.
        :type path_on_compute: str
        :param overwrite: Indicates whether to overwrite existing data.
        :type overwrite: bool
        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        raise NotImplementedError("SqlDataReference does not support `upload` operation.")

    def as_mount(self):
        """Switch data reference operation to mount. Not supported for SqlDataReference.

        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        raise NotImplementedError("SqlDataReference does not support `mount` operation.")

    def to_config(self):
        """Convert the DataReference object to DataReferenceConfiguration object. Not supported for SqlDataReference.

        :return: A new DataReferenceConfiguration object.
        :rtype: azureml.core.runconfig.DataReferenceConfiguration
        """
        raise NotImplementedError("SqlDataReference does not support `to_config` operation.")

    def _clone(self):
        return SqlDataReference(
            datastore=self.datastore,
            sql_table=self.sql_table,
            sql_query=self.sql_query,
            sql_stored_procedure=self.sql_stored_procedure,
            sql_stored_procedure_params=self.sql_stored_procedure_params
        )
