# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli.datastore.datastore_subgroup import DatastoreSubGroup
from azureml._cli.cli_command import command
from azureml._cli import argument

from azureml.data.datastore_client import _DatastoreClient


INCLUDE_SECRET = argument.Argument("include_secret", "--include-secret", "",
                                   help="Show the registered secret for the datastores.", default=False)
DATASTORE_NAME = argument.Argument("datastore_name", "--name", "-n", help="The datastore name.", required=True)


@command(
    subgroup_type=DatastoreSubGroup,
    command="list",
    short_description="List datastores in the workspace",
    argument_list=[
        INCLUDE_SECRET
    ])
def list_datastores_in_workspace(workspace=None, include_secret=False, logger=None):
    dss = _DatastoreClient.list(workspace)
    return list(map(lambda d: d._as_dict(not include_secret), dss))


@command(
    subgroup_type=DatastoreSubGroup,
    command="show",
    short_description="Show a single datastore by name",
    argument_list=[
        DATASTORE_NAME,
        INCLUDE_SECRET
    ])
def get_datastore(
        workspace=None,
        datastore_name=None,
        include_secret=False,
        logger=None):

    return _DatastoreClient.get(workspace, datastore_name)._as_dict(not include_secret)


@command(
    subgroup_type=DatastoreSubGroup,
    command="show-default",
    short_description="Show the workspace default datastore",
    argument_list=[
        INCLUDE_SECRET
    ])
def show_default_datastore(
        workspace=None,
        include_secret=None,
        logger=None):
    return _DatastoreClient.get_default(workspace)._as_dict(not include_secret)


@command(
    subgroup_type=DatastoreSubGroup,
    command="set-default",
    short_description="Set the workspace default datastore by name",
    argument_list=[
        DATASTORE_NAME
    ])
def set_default_datastore(
        workspace=None,
        datastore_name=None,
        logger=None):
    _DatastoreClient.set_default(workspace, datastore_name)


STORAGE_ACCOUNT_NAME = argument.Argument("account_name", "--account-name", "-a",
                                         help="The name of the storage account.", required=True)
CONTAINER_NAME = argument.Argument("container_name", "--container-name", "-c",
                                   help="The blob container name.", required=True)
SAS_TOKEN = argument.Argument("sas_token", "--sas-token", "", help="A SAS token for the blob container.")
ACCOUNT_KEY = argument.Argument("account_key", "--account-key", "-k", help="The storage account key.")
PROTOCOL = argument.Argument("protocol", "--protocol", "",
                             help="Protocol to use to connect to the blob container. If not specified, "
                             "defaults to https.", default="https")
STORAGE_ENDPOINT = argument.Argument("endpoint", "--endpoint", "",
                                     help="The endpoint of the storage account. Defaults to core.windows.net.",
                                     default="core.windows.net")
STORAGE_ACCOUNT_SUBSCRIPTION_ID = argument.Argument(
    "storage_account_subscription_id", "--storage-account-subscription-id", "",
    help="The subscription ID of the storage account."
)
STORAGE_ACCOUNT_RESOURCE_GROUP = argument.Argument(
    "storage_account_resource_group", "--storage-account-resource-group", "",
    help="The resource group of the storage account."
)
GRANT_WORKSPACE_ACCESS = argument.Argument(
    "grant_workspace_access", "--grant-workspace-msi-access", "",
    help="Defaults to False. Set it to True to access data behind virtual network from Machine Learning Studio. "
         "This makes data access from Machine Learning Studio use workspace managed identity for authentication, "
         "You have to be Owner or User Access Administrator of the storage to opt-in. Ask your administrator "
         "to configure it for you if you do not have the required permission. "
         "Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network",
    default=False
)


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-blob",
    short_description="Attach a blob storage datastore",
    argument_list=[
        DATASTORE_NAME,
        STORAGE_ACCOUNT_NAME,
        CONTAINER_NAME,
        SAS_TOKEN,
        ACCOUNT_KEY,
        PROTOCOL,
        STORAGE_ENDPOINT,
        INCLUDE_SECRET,
        GRANT_WORKSPACE_ACCESS,
        STORAGE_ACCOUNT_SUBSCRIPTION_ID,
        STORAGE_ACCOUNT_RESOURCE_GROUP
    ])
def attach_blob_container(
        workspace=None,
        datastore_name=None,
        account_name=None,
        container_name=None,
        sas_token=None,
        account_key=None,
        protocol=None,
        endpoint=None,
        include_secret=False,
        grant_workspace_access=False,
        storage_account_subscription_id=None,
        storage_account_resource_group=None,
        logger=None):
    sas_token = _strip_quotes(sas_token)
    account_key = _strip_quotes(account_key)
    return (_DatastoreClient
            .register_azure_blob_container(workspace, datastore_name, container_name, account_name,
                                           sas_token, account_key, protocol, endpoint,
                                           grant_workspace_access=grant_workspace_access,
                                           subscription_id=storage_account_subscription_id,
                                           resource_group=storage_account_resource_group)
            ._as_dict(not include_secret))


SHARE_NAME = argument.Argument("share_name", "--share-name", "-c", help="The file share name.", required=True)


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-file",
    short_description="Attach a file share datastore",
    argument_list=[
        DATASTORE_NAME,
        STORAGE_ACCOUNT_NAME,
        SHARE_NAME,
        SAS_TOKEN,
        ACCOUNT_KEY,
        PROTOCOL,
        STORAGE_ENDPOINT,
        INCLUDE_SECRET
    ])
def attach_file_share(
        workspace=None,
        datastore_name=None,
        account_name=None,
        share_name=None,
        sas_token=None,
        account_key=None,
        protocol=None,
        endpoint=None,
        include_secret=False,
        logger=None):
    sas_token = _strip_quotes(sas_token)
    account_key = _strip_quotes(account_key)
    return (_DatastoreClient
            .register_azure_file_share(workspace, datastore_name, share_name, account_name, sas_token, account_key,
                                       protocol, endpoint)
            ._as_dict(not include_secret))


ADLS_STORE_NAME = argument.Argument("store_name", "--store-name", "-c",  # -c??
                                    help="The ADLS store name.", required=True)
SP_TENANT_ID = argument.Argument("tenant_id", "--tenant-id", "",
                                 help="The service principal Tenant ID.", required=True)
CLIENT_ID = argument.Argument("client_id", "--client-id", "",
                              help="The service principal's client/application ID.", required=True)
CLIENT_SECRET = argument.Argument("client_secret", "--client-secret", "",
                                  help="The service principal's secret.", required=True)
ADLS_SUBSCRIPTION_ID = argument.Argument("adls_subscription_id", "--adls-subscription-id", "",
                                         help="The ID of the subscription the ADLS store belongs to.")
ADLS_RESOURCE_GROUP = argument.Argument("adls_resource_group", "--adls-resource-group", "",
                                        help="The resource group the ADLS store belongs to.")
ADLS_RESOURCE_URL = argument.Argument("resource_url", "--resource-url", "",
                                      help="Determines what operations will be performed on the data lake store.",
                                      default="https://datalake.azure.net/")
AUTHORITY_URL = argument.Argument("authority_url", "--authority-url", "",
                                  help="Authority url used to authenticate the user.",
                                  default="https://login.microsoftonline.com")


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-adls",
    short_description="Attach an ADLS datastore",
    argument_list=[
        DATASTORE_NAME,
        ADLS_STORE_NAME,
        SP_TENANT_ID,
        CLIENT_ID,
        CLIENT_SECRET,
        ADLS_SUBSCRIPTION_ID,
        ADLS_RESOURCE_GROUP,
        ADLS_RESOURCE_URL,
        AUTHORITY_URL,
        INCLUDE_SECRET,
        GRANT_WORKSPACE_ACCESS
    ])
def attach_adls(
        workspace=None,
        datastore_name=None,
        store_name=None,
        tenant_id=None,
        client_id=None,
        client_secret=None,
        adls_subscription_id=None,
        adls_resource_group=None,
        resource_url=None,
        authority_url=None,
        include_secret=False,
        logger=None,
        grant_workspace_access=False):
    client_secret = _strip_quotes(client_secret)
    return (_DatastoreClient
            .register_azure_data_lake(workspace, datastore_name, store_name, tenant_id, client_id, client_secret,
                                      resource_url=resource_url, authority_url=authority_url,
                                      subscription_id=adls_subscription_id,
                                      resource_group=adls_resource_group,
                                      grant_workspace_access=grant_workspace_access)
            ._as_dict(not include_secret))


FILE_SYSTEM = argument.Argument("file_system", "--file-system", "-c",
                                help="The file system name of the ADLS Gen2.", required=True)
ADLS_GEN2_RESOURCE_URL = argument.Argument("resource_url", "--resource-url", "",
                                           help="Determines what operations will be performed on the data lake store.",
                                           default="https://storage.azure.com/")
ADLS_GEN2_SUBSCRIPTION_ID = argument.Argument(
    "adlsgen2_subscription_id", "--adlsgen2-account-subscription-id", "",
    help="The subscription ID of the ADLS Gen2 storage account."
)
ADLS_GEN2_RESOURCE_GROUP = argument.Argument(
    "adlsgen2_resource_group", "--adlsgen2-account-resource-group", "",
    help="The resource group of the ADLS Gen2 storage account."
)


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-adls-gen2",
    short_description="Attach a ADLS Gen2 datastore",
    argument_list=[
        DATASTORE_NAME,
        STORAGE_ACCOUNT_NAME,
        FILE_SYSTEM,
        SP_TENANT_ID,
        CLIENT_ID,
        CLIENT_SECRET,
        PROTOCOL,
        STORAGE_ENDPOINT,
        ADLS_GEN2_RESOURCE_URL,
        AUTHORITY_URL,
        INCLUDE_SECRET,
        ADLS_GEN2_SUBSCRIPTION_ID,
        ADLS_GEN2_RESOURCE_GROUP,
        GRANT_WORKSPACE_ACCESS
    ])
def attach_adls_gen2(
        workspace=None,
        datastore_name=None,
        account_name=None,
        file_system=None,
        tenant_id=None,
        client_id=None,
        client_secret=None,
        protocol=None,
        endpoint=None,
        resource_url=None,
        authority_url=None,
        include_secret=False,
        logger=None,
        adlsgen2_subscription_id=None,
        adlsgen2_resource_group=None,
        grant_workspace_access=False):
    return (_DatastoreClient
            .register_azure_data_lake_gen2(workspace, datastore_name, file_system, account_name, protocol, endpoint,
                                           tenant_id, client_id, client_secret, resource_url, authority_url,
                                           subscription_id=adlsgen2_subscription_id,
                                           resource_group=adlsgen2_resource_group,
                                           grant_workspace_access=grant_workspace_access)
            ._as_dict(not include_secret))


SERVER_NAME = argument.Argument("server_name", "--server-name", "", help="The SQL/PostgreSQL/MySQL server name.",
                                required=True)
DATABASE_NAME = argument.Argument("database_name", "--database-name", "-d",
                                  help="The database name.", required=True)
SQL_ENDPOINT = argument.Argument("endpoint", "--endpoint", "",
                                 help="The endpoint of the sql server. Defaults to database.windows.net.",
                                 default="database.windows.net")
SQL_RESOURCE_URL = argument.Argument("resource_url", "--resource-url", "",
                                     help="Determines what operations will be performed on the database.",
                                     default="https://database.windows.net/")
USERNAME = argument.Argument("username", "--username", "",
                             help="The username of the database user to access the database.")
PASSWORD = argument.Argument("password", "--password", "",
                             help="The password of the database user to access the database.")
SQL_SP_TENANT_ID = argument.Argument("tenant_id", "--tenant-id", "", help="The service principal Tenant ID.")
SQL_CLIENT_ID = argument.Argument("client_id", "--client-id", "", help="The service principal/application ID.")
SQL_CLIENT_SECRET = argument.Argument("client_secret", "--client-secret", "", help="The service principal's secret.")
SQL_SUBSCRIPTION_ID = argument.Argument(
    "sql_subscription_id", "--sql-subscription-id", "",
    help="The subscription ID of the Azure Sql Server."
)
SQL_RESOURCE_GROUP = argument.Argument(
    "sql_resource_group", "--sql-resource-group", "",
    help="The resource group of the Azure Sql Server."
)


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-sqldb",
    short_description="Attach an Azure SQL datastore",
    argument_list=[
        DATASTORE_NAME,
        SERVER_NAME,
        DATABASE_NAME,
        SQL_SP_TENANT_ID,
        SQL_CLIENT_ID,
        SQL_CLIENT_SECRET,
        SQL_RESOURCE_URL,
        AUTHORITY_URL,
        SQL_ENDPOINT,
        INCLUDE_SECRET,
        USERNAME,
        PASSWORD,
        SQL_SUBSCRIPTION_ID,
        SQL_RESOURCE_GROUP,
        GRANT_WORKSPACE_ACCESS
    ])
def attach_sqldb(
        workspace=None,
        datastore_name=None,
        server_name=None,
        database_name=None,
        tenant_id=None,
        client_id=None,
        client_secret=None,
        resource_url=None,
        authority_url=None,
        endpoint=None,
        include_secret=False,
        username=None,
        password=None,
        logger=None,
        sql_subscription_id=None,
        sql_resource_group=None,
        grant_workspace_access=False):
    client_secret = _strip_quotes(client_secret)
    return (_DatastoreClient
            .register_azure_sql_database(workspace, datastore_name, server_name, database_name,
                                         tenant_id, client_id, client_secret,
                                         resource_url=resource_url, authority_url=authority_url, endpoint=endpoint,
                                         username=username, password=password,
                                         subscription_id=sql_subscription_id,
                                         resource_group=sql_resource_group,
                                         grant_workspace_access=grant_workspace_access)
            ._as_dict(not include_secret))


USER_ID = argument.Argument("user_id", "--user-id", "-u", help="The user ID.", required=True)
PASSWORD = argument.Argument("password", "--password", "-p", help="The password.", required=True)
PORT = argument.Argument("port", "--port", "", help="The port number", default="5432")
ENFORCE_SSL = argument.Argument("enforce_ssl", "--enforce-ssl", "",
                                help="This sets the ssl value of the server. Defaults to true if not set.",
                                default=True)
PSQL_ENDPOINT = argument.Argument("endpoint", "--endpoint", "",
                                  help="The endpoint of the server. Defaults to postgres.database.azure.com.",
                                  default="postgres.database.azure.com")


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-psqldb",
    short_description="Attach an Azure PostgreSQL datastore",
    argument_list=[
        DATASTORE_NAME,
        SERVER_NAME,
        DATABASE_NAME,
        USER_ID,
        PASSWORD,
        PORT,
        ENFORCE_SSL,
        PSQL_ENDPOINT,
        INCLUDE_SECRET
    ])
def attach_postgresqldb(
        workspace=None,
        datastore_name=None,
        server_name=None,
        database_name=None,
        user_id=None,
        password=None,
        port=None,
        enforce_ssl=True,
        endpoint=None,
        include_secret=False,
        logger=None):
    user_password = _strip_quotes(password)
    return (_DatastoreClient
            .register_azure_postgre_sql(workspace, datastore_name, server_name, database_name, user_id, user_password,
                                        port_number=port, enforce_ssl=enforce_ssl, endpoint=endpoint)
            ._as_dict(not include_secret))


MYSQL_ENDPOINT = argument.Argument("endpoint", "--endpoint", "",
                                   help="The endpoint of the server. Defaults to mysql.database.azure.com.",
                                   default="mysql.database.azure.com")


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-mysqldb",
    short_description="Attach an Azure MySQL datastore",
    argument_list=[
        DATASTORE_NAME,
        SERVER_NAME,
        DATABASE_NAME,
        USER_ID,
        PASSWORD,
        PORT,
        MYSQL_ENDPOINT,
        INCLUDE_SECRET
    ])
def attach_mysqldb(
        workspace=None,
        datastore_name=None,
        server_name=None,
        database_name=None,
        user_id=None,
        password=None,
        port=None,
        endpoint=None,
        include_secret=False,
        logger=None):
    user_password = _strip_quotes(password)
    return (_DatastoreClient
            .register_azure_my_sql(workspace, datastore_name, server_name, database_name, user_id, user_password,
                                   port_number=port, endpoint=endpoint)
            ._as_dict(not include_secret))


@command(
    subgroup_type=DatastoreSubGroup,
    command="attach-dbfs",
    short_description="Attach a Databricks File System datastore",
    argument_list=[
        DATASTORE_NAME
    ])
def attach_dbfs(
        workspace=None,
        datastore_name=None,
        logger=None):
    return _DatastoreClient.register_dbfs(workspace, datastore_name)._as_dict()


@command(
    subgroup_type=DatastoreSubGroup,
    command="detach",
    short_description="Detach a datastore by name",
    argument_list=[
        DATASTORE_NAME
    ])
def detach_datastore(
        workspace=None,
        datastore_name=None,
        logger=None):
    _DatastoreClient.delete(workspace, datastore_name)


DOWNLOAD_TARGET_PATH = argument.Argument("target_path", "--target-path", "-d",
                                         help="Target path for the downloaded files", required=True)
FILE_PREFIX = argument.Argument("prefix", "--prefix", "-p",
                                help="Path filter for files to download. If none is provided, downloads everything.")
OVERWRITE = argument.Argument("overwrite", "--overwrite", "",
                              help="Overwrite target files if they exist.", default=False)
HIDE_PROGRESS = argument.Argument("hide_progress", "--hide-progress", "", help="Whether to hide progress of operation",
                                  default=False)


@command(
    subgroup_type=DatastoreSubGroup,
    command="download",
    short_description="Download files from a Datastore",
    argument_list=[
        DATASTORE_NAME,
        DOWNLOAD_TARGET_PATH,
        FILE_PREFIX,
        OVERWRITE,
        HIDE_PROGRESS
    ])
def download(
        workspace=None,
        datastore_name=None,
        target_path=None,
        prefix=None,
        overwrite=False,
        hide_progress=False,
        logger=None):
    _DatastoreClient.get(workspace, datastore_name).download(target_path, prefix, overwrite, not hide_progress)


SRC_PATH = argument.Argument("src_path", "--src-path", "-p", help="Path from which to upload data.", required=True)
UPLOAD_TARGET_PATH = argument.Argument("target_path", "--target-path", "-u",
                                       help="Path to upload data in the container. Uploads to the root by default.")


@command(
    subgroup_type=DatastoreSubGroup,
    command="upload",
    short_description="Upload files to a Datastore",
    argument_list=[
        DATASTORE_NAME,
        SRC_PATH,
        UPLOAD_TARGET_PATH,
        OVERWRITE,
        HIDE_PROGRESS
    ])
def upload(
        workspace=None,
        datastore_name=None,
        src_path=None,
        target_path=None,
        overwrite=False,
        hide_progress=False,
        logger=None):
    _DatastoreClient.get(workspace, datastore_name).upload(src_path, target_path, overwrite, not hide_progress)


def _strip_quotes(val):
    return val.strip('"') if val else val
