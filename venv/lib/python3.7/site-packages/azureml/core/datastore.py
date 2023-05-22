# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing Datastores in Azure Machine Learning."""

from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml.exceptions import UserErrorException


class Datastore(object):
    """Represents a storage abstraction over an Azure Machine Learning storage account.

    Datastores are attached to workspaces and are used to store connection information to Azure
    storage services so you can refer to them by name and don't need to remember the connection
    information and secret used to connect to the storage services.

    Examples of supported Azure storage services that can be registered as datastores are:

    * Azure Blob Container
    * Azure File Share
    * Azure Data Lake
    * Azure Data Lake Gen2
    * Azure SQL Database
    * Azure Database for PostgreSQL
    * Databricks File System
    * Azure Database for MySQL

    Use this class to perform management operations, including register, list, get, and remove datastores.
    Datastores for each service are created with the ``register*`` methods of this class. When using a datastore
    to access data, you must have permission to access that data, which depends on the credentials registered
    with the datastore.

    For more information on datastores and how they can be used in machine learning see the following articles:

    * `Access data in Azure storage
      services <https://docs.microsoft.com/azure/machine-learning/how-to-access-data>`_
    * `Train models with Azure Machine Learning using
      estimator <https://docs.microsoft.com/azure/machine-learning/how-to-train-ml-models>`_
    * `Create and run machine learning
      pipelines <https://docs.microsoft.com/azure/machine-learning/how-to-create-your-first-pipeline>`_

    .. remarks::

        To interact with data in your datastores for machine learning tasks, like training, `create an Azure Machine
        Learning dataset <https://aka.ms/azureml/howto/createdatasets>`_.
        Datasets provide functions that load tabular data into a pandas or Spark DataFrame.
        Datasets also provide the ability to download or mount files of any format from Azure Blob storage,
        Azure Files, Azure Data Lake Storage Gen1, Azure Data Lake Storage Gen2, Azure SQL Database, and Azure Database
        for PostgreSQL. `Learn more about how to train with datasets
        <https://aka.ms/azureml/howto/trainwithdatasets>`_.

        The following example shows how to create a Datastore connected to Azure Blob Container.

        .. code-block:: python

            from azureml.exceptions import UserErrorException

            blob_datastore_name='MyBlobDatastore'
            account_name=os.getenv("BLOB_ACCOUNTNAME_62", "<my-account-name>") # Storage account name
            container_name=os.getenv("BLOB_CONTAINER_62", "<my-container-name>") # Name of Azure blob container
            account_key=os.getenv("BLOB_ACCOUNT_KEY_62", "<my-account-key>") # Storage account key

            try:
                blob_datastore = Datastore.get(ws, blob_datastore_name)
                print("Found Blob Datastore with name: %s" % blob_datastore_name)
            except UserErrorException:
                blob_datastore = Datastore.register_azure_blob_container(
                    workspace=ws,
                    datastore_name=blob_datastore_name,
                    account_name=account_name, # Storage account name
                    container_name=container_name, # Name of Azure blob container
                    account_key=account_key) # Storage account key
                print("Registered blob datastore with name: %s" % blob_datastore_name)

            blob_data_ref = DataReference(
                datastore=blob_datastore,
                data_reference_name="blob_test_data",
                path_on_datastore="testdata")

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-data-transfer.ipynb


    """

    def __new__(cls, workspace, name=None):
        """Get a datastore by name. This call will make a request to the datastore service.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datastore, defaults to None, which gets the default datastore.
        :type name: str, optional
        :return: The corresponding datastore for that name.
        :rtype: AbstractDatastore
        """
        if name is None:
            return Datastore._client().get_default(workspace)
        return Datastore._client().get(workspace, name)

    def __init__(self, workspace, name=None):
        """Get a datastore by name. This call will make a request to the datastore service.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datastore, defaults to None, which gets the default datastore.
        :type name: str, optional
        :return: The corresponding datastore for that name.
        :rtype: AbstractDatastore
        """
        self.workspace = workspace
        self.name = name

    def set_as_default(self):
        """Set the default datastore.

        :param datastore_name: The name of the datastore.
        :type datastore_name: str
        """
        Datastore._client().set_default(self.workspace, self.name)

    def unregister(self):
        """Unregisters the datastore. the underlying storage service will not be deleted."""
        Datastore._client().delete(self.workspace, self.name)

    @staticmethod
    def get(workspace, datastore_name):
        """Get a datastore by name. This is same as calling the constructor.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The name of the datastore, defaults to None, which gets the default datastore.
        :type datastore_name: str, optional
        :return: The corresponding datastore for that name.
        :rtype: azureml.data.azure_storage_datastore.AzureFileDatastore
                or azureml.data.azure_storage_datastore.AzureBlobDatastore
                or azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore
                or azureml.data.azure_data_lake_datastore.AzureDataLakeGen2Datastore
                or azureml.data.azure_sql_database_datastore.AzureSqlDatabaseDatastore
                or azureml.data.azure_postgre_sql_datastore.AzurePostgreSqlDatastore
                or azureml.data.azure_my_sql_datastore.AzureMySqlDatastore
                or azureml.data.dbfs_datastore.DBFSDatastore
        """
        return Datastore._client().get(workspace, datastore_name)

    @staticmethod
    def get_default(workspace):
        """Get the default datastore for the workspace.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :return: The default datastore for the workspace
        :rtype: azureml.data.azure_storage_datastore.AzureFileDatastore
                or azureml.data.azure_storage_datastore.AzureBlobDatastore
        """
        return Datastore._client().get_default(workspace)

    @staticmethod
    def register_azure_blob_container(workspace, datastore_name, container_name, account_name, sas_token=None,
                                      account_key=None, protocol=None, endpoint=None, overwrite=False,
                                      create_if_not_exists=False, skip_validation=False, blob_cache_timeout=None,
                                      grant_workspace_access=False, subscription_id=None, resource_group=None):
        """Register an Azure Blob Container to the datastore.

        Credential based (GA) and identity based (Preview) data access are supported, you can choose to use SAS Token
        or Storage Account Key. If no credential is saved with the datastore,
        users' AAD token will be used in notebook or local python program if it directly calls one of these functions:
        FileDataset.mount
        FileDataset.download
        FileDataset.to_path
        TabularDataset.to_pandas_dataframe
        TabularDataset.to_dask_dataframe
        TabularDataset.to_spark_dataframe
        TabularDataset.to_parquet_files
        TabularDataset.to_csv_files
        the identity of the compute target will be used in jobs submitted by Experiment.submit
        for data access authentication. `Learn more here <https://aka.ms/data-access>`_.

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The name of the datastore, case insensitive, can only contain alphanumeric characters
            and _.
        :type datastore_name: str
        :param container_name: The name of the azure blob container.
        :type container_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param sas_token: An account SAS token, defaults to None. For data read, we require a minimum of List & Read
            permissions for Containers & Objects and for data write we additionally require Write & Add permissions.
        :type sas_token: str, optional
        :param account_key: Access keys of your storage account, defaults to None.
        :type account_key: str, optional
        :param protocol: Protocol to use to connect to the blob container. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the storage account. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
            it will create one, defaults to False
        :type overwrite: bool, optional
        :param create_if_not_exists: create the blob container if it does not exists, defaults to False
        :type create_if_not_exists: bool, optional
        :param skip_validation: skips validation of storage keys, defaults to False
        :type skip_validation: bool, optional
        :param blob_cache_timeout: When this blob is mounted, set the cache timeout to this many seconds.
            If None, defaults to no timeout (i.e. blobs will be cached for the duration of the job when read).
        :type blob_cache_timeout: int, optional
        :param grant_workspace_access: Defaults to False. Set it to True to access data behind virtual network
            from Machine Learning Studio.This makes data access from Machine Learning Studio use workspace managed
            identity for authentication, and adds the workspace managed identity as Reader of the storage.
            You have to be owner or user access administrator of the storage to opt-in. Ask your administrator
            to configure it for you if you do not have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        :param subscription_id: The subscription id of the storage account, defaults to None.
        :type subscription_id: str, optional
        :param resource_group: The resource group of the storage account, defaults to None.
        :type resource_group: str, optional
        :return: The blob datastore.
        :rtype: azureml.data.azure_storage_datastore.AzureBlobDatastore
        """
        return Datastore._client().register_azure_blob_container(
            workspace=workspace,
            datastore_name=datastore_name,
            container_name=container_name,
            account_name=account_name,
            sas_token=sas_token,
            account_key=account_key,
            protocol=protocol,
            endpoint=endpoint,
            overwrite=overwrite,
            create_if_not_exists=create_if_not_exists,
            skip_validation=skip_validation,
            blob_cache_timeout=blob_cache_timeout,
            grant_workspace_access=grant_workspace_access,
            subscription_id=subscription_id,
            resource_group=resource_group,
        )

    @staticmethod
    def register_azure_file_share(workspace, datastore_name, file_share_name, account_name, sas_token=None,
                                  account_key=None, protocol=None, endpoint=None, overwrite=False,
                                  create_if_not_exists=False, skip_validation=False):
        """Register an Azure File Share to the datastore.

        You can choose to use SAS Token or Storage Account Key

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The name of the datastore, case insensitive, can only contain alphanumeric characters
            and _.
        :type datastore_name: str
        :param file_share_name: The name of the azure file container.
        :type file_share_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param sas_token: An account SAS token, defaults to None. For data read, we require a minimum of List & Read
            permissions for Containers & Objects and for data write we additionally require Write & Add permissions.
        :type sas_token: str, optional
        :param account_key: Access keys of your storage account, defaults to None.
        :type account_key: str, optional
        :param protocol: The protocol to use to connect to the file share. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the file share. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param overwrite: Whether to overwrite an existing datastore. If the datastore does not exist,
            it will create one. The default is False.
        :type overwrite: bool, optional
        :param create_if_not_exists: Whether to create the file share if it does not exists. The default is False.
        :type create_if_not_exists: bool, optional
        :param skip_validation: Whether to skip validation of storage keys. The default is False.
        :type skip_validation: bool, optional
        :return: The file datastore.
        :rtype: azureml.data.azure_storage_datastore.AzureFileDatastore
        """
        return Datastore._client().register_azure_file_share(workspace, datastore_name, file_share_name, account_name,
                                                             sas_token, account_key, protocol, endpoint, overwrite,
                                                             create_if_not_exists, skip_validation)

    @staticmethod
    def register_azure_data_lake(workspace, datastore_name, store_name, tenant_id=None, client_id=None,
                                 client_secret=None, resource_url=None, authority_url=None, subscription_id=None,
                                 resource_group=None, overwrite=False, grant_workspace_access=False):
        r"""Initialize a new Azure Data Lake Datastore.

        Credential based (GA) and identity based (Preview) data access are supported,
        You can register a datastore with Service Principal for credential based data access.
        If no credential is saved with the datastore,
        users' AAD token will be used in notebook or local python program if it directly calls one of these functions:
        FileDataset.mount
        FileDataset.download
        FileDataset.to_path
        TabularDataset.to_pandas_dataframe
        TabularDataset.to_dask_dataframe
        TabularDataset.to_spark_dataframe
        TabularDataset.to_parquet_files
        TabularDataset.to_csv_files
        the identity of the compute target will be used in jobs submitted by Experiment.submit
        for data access authentication. `Learn more here <https://aka.ms/data-access>`_.

        Please see below for an example of how to register an Azure Data Lake Gen1 as a Datastore.

        .. code-block:: python

            adlsgen1_datastore_name='adlsgen1datastore'

            store_name=os.getenv("ADL_STORENAME", "<my_datastore_name>") # the ADLS name
            subscription_id=os.getenv("ADL_SUBSCRIPTION", "<my_subscription_id>") # subscription id of the ADLS
            resource_group=os.getenv("ADL_RESOURCE_GROUP", "<my_resource_group>") # resource group of ADLS
            tenant_id=os.getenv("ADL_TENANT", "<my_tenant_id>") # tenant id of service principal
            client_id=os.getenv("ADL_CLIENTID", "<my_client_id>") # client id of service principal
            client_secret=os.getenv("ADL_CLIENT_SECRET", "<my_client_secret>") # the secret of service principal

            adls_datastore = Datastore.register_azure_data_lake(
                workspace=ws,
                datastore_name=aslsgen1_datastore_name,
                subscription_id=subscription_id, # subscription id of ADLS account
                resource_group=resource_group, # resource group of ADLS account
                store_name=store_name, # ADLS account name
                tenant_id=tenant_id, # tenant id of service principal
                client_id=client_id, # client id of service principal
                client_secret=client_secret) # the secret of service principal

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

            .. note::

                Azure Data Lake Datastore supports data transfer and running U-Sql jobs using \
                Azure Machine Learning Pipelines.

                You can also use it as a data source for Azure Machine Learning Dataset which can be downloaded \
                or mounted on any supported compute.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The datastore name.
        :type datastore_name: str
        :param store_name: The ADLS store name.
        :type store_name: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal used to access data.
        :type tenant_id: str, optional
        :param client_id: The Client ID/Application ID of the service principal used to access data.
        :type client_id: str, optional
        :param client_secret: The Client Secret of the service principal used to access data.
        :type client_secret: str, optional
        :param resource_url: The resource URL, which determines what operations will be performed on the Data Lake
            store, if None, defaults to ``https://datalake.azure.net/`` which allows us to perform filesystem
            operations.
        :type resource_url: str, optional
        :param authority_url: The authority URL used to authenticate the user, defaults to
            ``https://login.microsoftonline.com``.
        :type authority_url: str, optional
        :param subscription_id: The ID of the subscription the ADLS store belongs to.
        :type subscription_id: str, optional
        :param resource_group: The resource group the ADLS store belongs to.
        :type resource_group: str, optional
        :param overwrite: Whether to overwrite an existing datastore. If the datastore does not exist,
            it will create one. The default is False.
        :type overwrite: bool, optional
        :return: Returns the Azure Data Lake Datastore.
        :rtype: azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore
        :param grant_workspace_access: Defaults to False. Set it to True to access data behind virtual network
            from Machine Learning Studio.This makes data access from Machine Learning Studio use workspace managed
            identity for authentication, and adds the workspace managed identity as Reader of the storage.
            You have to be Owner or User Access Administrator of the storage to opt-in. Ask your administrator
            to configure it for you if you do not have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        """
        return Datastore._client().register_azure_data_lake(
            workspace=workspace,
            datastore_name=datastore_name,
            store_name=store_name,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            resource_url=resource_url,
            authority_url=authority_url,
            subscription_id=subscription_id,
            resource_group=resource_group,
            overwrite=overwrite,
            grant_workspace_access=grant_workspace_access)

    @staticmethod
    def register_azure_data_lake_gen2(workspace, datastore_name, filesystem, account_name, tenant_id=None,
                                      client_id=None, client_secret=None, resource_url=None, authority_url=None,
                                      protocol=None, endpoint=None, overwrite=False, subscription_id=None,
                                      resource_group=None, grant_workspace_access=False):
        """Initialize a new Azure Data Lake Gen2 Datastore.

        Credential based (GA) and identity based (Preview) data access are supported,
        You can register a datastore with Service Principal for credential based data access.
        If no credential is saved with the datastore,
        users' AAD token will be used in notebook or local python program if it directly calls one of these functions:
        FileDataset.mount
        FileDataset.download
        FileDataset.to_path
        TabularDataset.to_pandas_dataframe
        TabularDataset.to_dask_dataframe
        TabularDataset.to_spark_dataframe
        TabularDataset.to_parquet_files
        TabularDataset.to_csv_files
        the identity of the compute target will be used in jobs submitted by Experiment.submit
        for data access authentication. `Learn more here <https://aka.ms/data-access>`_.

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The datastore name.
        :type datastore_name: str
        :param filesystem: The name of the Data Lake Gen2 filesystem.
        :type filesystem: str
        :param account_name: The storage account name.
        :type account_name: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal.
        :type tenant_id: str, optional
        :param client_id: The Client ID/Application ID of the service principal.
        :type client_id: str, optional
        :param client_secret: The secret of the service principal.
        :type client_secret: str, optional
        :param resource_url: The resource URL, which determines what operations will be performed on
            the data lake store, defaults to ``https://storage.azure.com/`` which allows us to perform filesystem
            operations.
        :type resource_url: str, optional
        :param authority_url: The authority URL used to authenticate the user, defaults to
            ``https://login.microsoftonline.com``.
        :type authority_url: str, optional
        :param protocol: Protocol to use to connect to the blob container. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the storage account. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param overwrite: Whether to overwrite an existing datastore. If the datastore does not exist,
            it will create one. The default is False.
        :type overwrite: bool, optional
        :param subscription_id: The ID of the subscription the ADLS store belongs to.
        :type subscription_id: str, optional
        :param resource_group: The resource group the ADLS store belongs to.
        :type resource_group: str, optional
        :param grant_workspace_access: Defaults to False. Set it to True to access data behind virtual network
            from Machine Learning Studio.This makes data access from Machine Learning Studio use workspace managed
            identity for authentication, and adds the workspace managed identity as Reader of the storage.
            You have to be owner or user access administrator of the storage to opt-in. Ask your administrator
            to configure it for you if you do not have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        :return: Returns the Azure Data Lake Gen2 Datastore.
        :rtype: azureml.data.azure_data_lake_datastore.AzureDataLakeGen2Datastore

        """
        return Datastore._client().register_azure_data_lake_gen2(
            workspace=workspace,
            datastore_name=datastore_name,
            container_name=filesystem,
            account_name=account_name,
            tenant_id=tenant_id,
            client_id=client_id,
            protocol=protocol,
            endpoint=endpoint,
            client_secret=client_secret,
            resource_url=resource_url,
            authority_url=authority_url,
            overwrite=overwrite,
            subscription_id=subscription_id,
            resource_group=resource_group,
            grant_workspace_access=grant_workspace_access)

    @staticmethod
    def register_azure_sql_database(workspace, datastore_name, server_name, database_name, tenant_id=None,
                                    client_id=None, client_secret=None, resource_url=None, authority_url=None,
                                    endpoint=None, overwrite=False, username=None, password=None, subscription_id=None,
                                    resource_group=None, grant_workspace_access=False, **kwargs):
        """Initialize a new Azure SQL database Datastore.

        Credential based (GA) and identity based (Preview) data access are supported,
        you can choose to use Service Principal or username + password. If no credential is saved with the datastore,
        users' AAD token will be used in notebook or local python program if it directly calls one of these functions:
        FileDataset.mount
        FileDataset.download
        FileDataset.to_path
        TabularDataset.to_pandas_dataframe
        TabularDataset.to_dask_dataframe
        TabularDataset.to_spark_dataframe
        TabularDataset.to_parquet_files
        TabularDataset.to_csv_files
        the identity of the compute target will be used in jobs submitted by Experiment.submit
        for data access authentication. `Learn more here <https://aka.ms/data-access>`_.

        Please see below for an example of how to register an Azure SQL database as a Datastore.

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

            .. code-block:: python

                sql_datastore_name="azuresqldatastore"
                server_name=os.getenv("SQL_SERVERNAME", "<my_server_name>") # Name of the Azure SQL server
                database_name=os.getenv("SQL_DATABASENAME", "<my_database_name>") # Name of the Azure SQL database
                username=os.getenv("SQL_USER_NAME", "<my_sql_user_name>") # The username of the database user.
                password=os.getenv("SQL_USER_PASSWORD", "<my_sql_user_password>") # The password of the database user.

                sql_datastore = Datastore.register_azure_sql_database(
                    workspace=ws,
                    datastore_name=sql_datastore_name,
                    server_name=server_name,  # name should not contain fully qualified domain endpoint
                    database_name=database_name,
                    username=username,
                    password=password,
                    endpoint='database.windows.net')

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The datastore name.
        :type datastore_name: str
        :param server_name: The SQL server name. For fully qualified domain name like "sample.database.windows.net",
            the server_name value should be "sample", and the endpoint value should be "database.windows.net".
        :type server_name: str
        :param database_name: The SQL database name.
        :type database_name: str
        :param tenant_id: The Directory ID/Tenant ID of the service principal.
        :type tenant_id: str
        :param client_id: The Client ID/Application ID of the service principal.
        :type client_id: str
        :param client_secret: The secret of the service principal.
        :type client_secret: str
        :param resource_url: The resource URL, which determines what operations will be performed on
            the SQL database store, if None, defaults to https://database.windows.net/.
        :type resource_url: str, optional
        :param authority_url: The authority URL used to authenticate the user, defaults to
            https://login.microsoftonline.com.
        :type authority_url: str, optional
        :param endpoint: The endpoint of the SQL server. If None, defaults to database.windows.net.
        :type endpoint: str, optional
        :param overwrite: Whether to overwrite an existing datastore. If the datastore does not exist,
            it will create one. The default is False.
        :type overwrite: bool, optional
        :param username: The username of the database user to access the database.
        :type username: str
        :param password: The password of the database user to access the database.
        :type password: str
        :param skip_validation: Whether to skip validation of connecting to the SQL database. Defaults to False.
        :type skip_validation: bool, optional
        :param subscription_id: The ID of the subscription the ADLS store belongs to.
        :type subscription_id: str, optional
        :param resource_group: The resource group the ADLS store belongs to.
        :type resource_group: str, optional
        :param grant_workspace_access: Defaults to False. Set it to True to access data behind virtual network
            from Machine Learning Studio.This makes data access from Machine Learning Studio use workspace managed
            identity for authentication, and adds the workspace managed identity as Reader of the storage.
            You have to be owner or user access administrator of the storage to opt-in. Ask your administrator
            to configure it for you if you do not have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        :return: Returns the SQL database Datastore.
        :rtype: azureml.data.azure_sql_database_datastore.AzureSqlDatabaseDatastore
        """
        return Datastore._client().register_azure_sql_database(
            workspace=workspace,
            datastore_name=datastore_name,
            server_name=server_name,
            database_name=database_name,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            resource_url=resource_url,
            authority_url=authority_url,
            endpoint=endpoint,
            overwrite=overwrite,
            username=username,
            password=password,
            subscription_id=subscription_id,
            resource_group=resource_group,
            grant_workspace_access=grant_workspace_access, **kwargs)

    @ staticmethod
    def register_azure_postgre_sql(workspace, datastore_name, server_name, database_name, user_id, user_password,
                                   port_number=None, endpoint=None,
                                   overwrite=False, enforce_ssl=True, **kwargs):
        """Initialize a new Azure PostgreSQL Datastore.

        Please see below for an example of how to register an Azure PostgreSQL database as a Datastore.

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

            .. code-block:: python

                psql_datastore_name="postgresqldatastore"
                server_name=os.getenv("PSQL_SERVERNAME", "<my_server_name>") # FQDN name of the PostgreSQL server
                database_name=os.getenv("PSQL_DATBASENAME", "<my_database_name>") # Name of the PostgreSQL database
                user_id=os.getenv("PSQL_USERID", "<my_user_id>") # The database user id
                user_password=os.getenv("PSQL_USERPW", "<my_user_password>") # The database user password

                psql_datastore = Datastore.register_azure_postgre_sql(
                    workspace=ws,
                    datastore_name=psql_datastore_name,
                    server_name=server_name,
                    database_name=database_name,
                    user_id=user_id,
                    user_password=user_password)

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The datastore name.
        :type datastore_name: str
        :param server_name: The PostgreSQL server name.
        :type server_name: str
        :param database_name: The PostgreSQL database name.
        :type database_name: str
        :param user_id: The User ID of the PostgreSQL server.
        :type user_id: str
        :param user_password: The User Password of the PostgreSQL server.
        :type user_password: str
        :param port_number: The Port Number of the PostgreSQL server
        :type port_number: str
        :param endpoint: The endpoint of the PostgreSQL server. If None, defaults to postgres.database.azure.com.
        :type endpoint: str, optional
        :param overwrite: Whether to overwrite an existing datastore. If the datastore does not exist,
            it will create one. The default is False.
        :type overwrite: bool, optional
        :param enforce_ssl: Indicates SSL requirement of PostgreSQL server. Defaults to True.
        :type enforce_ssl: bool
        :return: Returns the PostgreSQL database Datastore.
        :rtype: azureml.data.azure_postgre_sql_datastore.AzurePostgreSqlDatastore
        """
        if user_id is None:
            raise UserErrorException("User id is required for PostreSQL.")

        if user_password is None:
            raise UserErrorException("User password is required for PostreSQL.")

        if server_name is None:
            raise UserErrorException("Server name is required for PostreSQL.")

        if database_name is None:
            raise UserErrorException("Database name is required for PostreSQL.")

        return Datastore._client().register_azure_postgre_sql(
            workspace, datastore_name, server_name, database_name, user_id, user_password,
            port_number, endpoint, overwrite, enforce_ssl, **kwargs)

    @ staticmethod
    def register_azure_my_sql(workspace, datastore_name, server_name, database_name, user_id, user_password,
                              port_number=None, endpoint=None, overwrite=False, **kwargs):
        """Initialize a new Azure MySQL Datastore.

        MySQL datastore can only be used to create DataReference as input and output to DataTransferStep in
        Azure Machine Learning pipelines. `More details can be found here <https://docs.microsoft.com/python/
        api/azureml-pipeline-steps/azureml.pipeline.steps.datatransferstep?view=azure-ml-py>`_.

        Please see below for an example of how to register an Azure MySQL database as a Datastore.

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

            .. code-block:: python

                mysql_datastore_name="mysqldatastore"
                server_name=os.getenv("MYSQL_SERVERNAME", "<my_server_name>") # FQDN name of the MySQL server
                database_name=os.getenv("MYSQL_DATBASENAME", "<my_database_name>") # Name of the MySQL database
                user_id=os.getenv("MYSQL_USERID", "<my_user_id>") # The User ID of the MySQL server
                user_password=os.getenv("MYSQL_USERPW", "<my_user_password>") # The user password of the MySQL server.

                mysql_datastore = Datastore.register_azure_my_sql(
                    workspace=ws,
                    datastore_name=mysql_datastore_name,
                    server_name=server_name,
                    database_name=database_name,
                    user_id=user_id,
                    user_password=user_password)

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The datastore name.
        :type datastore_name: str
        :param server_name: The MySQL server name.
        :type server_name: str
        :param database_name: The MySQL database name.
        :type database_name: str
        :param user_id: The User ID of the MySQL server.
        :type user_id: str
        :param user_password: The user password of the MySQL server.
        :type user_password: str
        :param port_number: The port number of the MySQL server.
        :type port_number: str
        :param endpoint: The endpoint of the MySQL server. If None, defaults to mysql.database.azure.com.
        :type endpoint: str, optional
        :param overwrite: Whether to overwrite an existing datastore. If the datastore does not exist,
            it will create one. The default is False.
        :type overwrite: bool, optional
        :return: Returns the MySQL database Datastore.
        :rtype: azureml.data.azure_my_sql_datastore.AzureMySqlDatastore
        """
        if user_id is None:
            raise UserErrorException("User id is required for My SQL.")

        if user_password is None:
            raise UserErrorException("User password is required for My SQL.")

        if server_name is None:
            raise UserErrorException("Server name is required for My SQL.")

        if database_name is None:
            raise UserErrorException("Database name is required for My SQL.")

        return Datastore._client().register_azure_my_sql(
            workspace, datastore_name, server_name, database_name, user_id, user_password,
            port_number, endpoint, overwrite, **kwargs)

    @staticmethod
    def register_dbfs(workspace, datastore_name):
        """Initialize a new Databricks File System (DBFS) datastore.

        The DBFS datastore can only be used to create DataReference as input and PipelineData as output to
        DatabricksStep in Azure Machine Learning pipelines.  `More details can be found here. <https://docs.
        microsoft.com/python/api/azureml-pipeline-steps/azureml.pipeline.steps.databricks_step.databricksstep
        ?view=azure-ml-py>`_.

        .. remarks::

            If you are attaching storage from different region than workspace region,
            it can result in higher latency and additional network usage costs.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: azureml.core.Workspace
        :param datastore_name: The datastore name.
        :type datastore_name: str
        :return: Returns the DBFS Datastore.
        :rtype: azureml.data.dbfs_datastore.DBFSDatastore
        """
        return Datastore._client().register_dbfs(workspace, datastore_name)

    @staticmethod
    @experimental
    def register_hdfs(workspace, datastore_name, protocol, namenode_address, hdfs_server_certificate,
                      kerberos_realm, kerberos_kdc_address, kerberos_principal, kerberos_keytab=None,
                      kerberos_password=None, overwrite=False):
        """Initialize a new HDFS datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param protocol: The protocol to use when communicating with the HDFS cluster. http or https.
            Possible values include: 'http', 'https'
        :type protocol: str or ~_restclient.models.enum
        :param namenode_address: The IP address or DNS hostname of the HDFS namenode. Optionally includes a port.
        :type namenode_address: str
        :param hdfs_server_certificate: The path to the TLS signing certificate of the HDFS namenode,
            if using TLS with a self-signed cert.
        :type hdfs_server_certificate: str, optional
        :param kerberos_realm: The Kerberos realm.
        :type kerberos_realm: str
        :param kerberos_kdc_address: The IP address or DNS hostname of the Kerberos KDC.
        :type kerberos_kdc_address: str
        :param kerberos_principal: The Kerberos principal to use for authentication and authorization.
        :type kerberos_principal: str
        :param kerberos_keytab: The path to the keytab file containing the key(s) corresponding to the
            Kerberos principal. Provide either this, or a password.
        :type kerberos_keytab: str, optional
        :param kerberos_password: The password corresponding to the Kerberos principal.
            Provide either this, or the path to a keytab file.
        :type kerberos_password: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
            it will create one. Defaults to False.
        :type overwrite: bool, optional
        """
        return Datastore._client().register_hdfs(workspace, datastore_name, protocol, namenode_address,
                                                 hdfs_server_certificate, kerberos_realm, kerberos_kdc_address,
                                                 kerberos_principal, kerberos_keytab, kerberos_password, overwrite)

    @staticmethod
    def _client():
        """Get a client.

        :return: Returns the client
        :rtype: DatastoreClient
        """
        from azureml.data.datastore_client import _DatastoreClient
        return _DatastoreClient
