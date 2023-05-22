# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Internal use only."""
import logging
import os
import re
import json
import uuid

from azureml._project import _commands
from azureml.exceptions import UserErrorException
from msrest.authentication import BasicTokenAuthentication
from msrest.exceptions import HttpOperationError
from azureml._base_sdk_common.service_discovery import get_service_url
from azureml._base_sdk_common import _ClientSessionId
from azureml._restclient.rest_client import RestClient
from azureml._restclient.models.data_store import DataStore
from azureml._restclient.models.azure_storage import AzureStorage
from azureml._restclient.models.azure_data_lake import AzureDataLake
from azureml._restclient.models.azure_sql_database import AzureSqlDatabase
from azureml._restclient.models.azure_postgre_sql import AzurePostgreSql
from azureml._restclient.models.azure_my_sql import AzureMySql
from azureml._restclient.models.on_prem_hdfs import OnPremHdfs
from azureml._restclient.models.client_credentials import ClientCredentials
from azureml._restclient.models.rest_client_enums import HdfsCredentialType, ServiceDataAccessAuthIdentity
from .azure_storage_datastore import AzureBlobDatastore, AzureFileDatastore
from .azure_data_lake_datastore import AzureDataLakeDatastore, AzureDataLakeGen2Datastore
from .azure_sql_database_datastore import AzureSqlDatabaseDatastore
from .azure_postgre_sql_datastore import AzurePostgreSqlDatastore
from .azure_my_sql_datastore import AzureMySqlDatastore
from .dbfs_datastore import DBFSDatastore
from .hdfs_datastore import HDFSDatastore
from .abstract_datastore import AbstractDatastore
from .constants import CONFLICT_MESSAGE, ROLE_ID_OF_READER, ROLE_ID_OF_STORAGE_BLOB_DATA_READER
from ._exception_handler import _handle_error_response_exception
from urllib.parse import urljoin, quote

module_logger = logging.getLogger(__name__)


class _DatastoreClient:
    """A client that provides methods to communicate with the datastore service."""

    # the auth token received from _auth.get_authentication_header is prefixed
    # with 'Bearer '. This is used to remove that prefix.
    _bearer_prefix_len = 7

    # list does not include the credential, we use this to substitute a fake account key
    # so we can create an azure storage sdk service object
    _account_key_substitute = "null"

    # the unique id of each python kernel process on client side to
    # correlate events within each process.
    _custom_headers = {"x-ms-client-session-id": _ClientSessionId}

    _capital_regex = re.compile("[A-Z]+")

    @staticmethod
    def get(workspace, datastore_name):
        """Get a datastore by name.

        :param workspace: The workspace.
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: The name of the datastore.
        :type datastore_name: str
        :return: The corresponding datastore for that name.
        :rtype: typing.Union[AzureFileDatastore, AzureBlobDatastore]
        """
        return _DatastoreClient._get(workspace, datastore_name)

    @staticmethod
    @_handle_error_response_exception
    def _grant_workspace_access(workspace, role, datastore_url, datastore_type, principal_id):
        rest_client = _DatastoreClient._get_client(workspace, workspace._auth, None)
        role_assignment = str(uuid.uuid4())
        url = datastore_url + '/' + role_assignment

        # Construct parameters
        query_parameters = {'api-version': '2018-07-01'}

        # Construct headers
        header_parameters = {'Content-Type': 'application/json',
                             'Authentication': 'Bearer ' + workspace._auth._get_arm_token()}

        # Construct body
        properties = {"roleDefinitionId": "/subscriptions/" + workspace.subscription_id
                                          + "/providers/Microsoft.Authorization/roleDefinitions/"
                                          + role,
                      "principalId": principal_id}
        data = {"Properties": properties}

        # Construct and send request
        request = rest_client._client.put(url, params=query_parameters)
        response = rest_client._client.send(request, header_parameters, data, stream=False)

        if response.status_code not in [200, 201, 409]:
            role_name = "'Reader ({})'".format(role) if role == ROLE_ID_OF_READER \
                else ("'Storage Blob Data Reader ({})'".format(role) if role == ROLE_ID_OF_STORAGE_BLOB_DATA_READER
                      else "'Undefined ({})'".format(role))
            error_message = ("Failed to grant workspace MSI permission to access the storage. "
                             "Please contact an owner of the storage account to manually assign the "
                             "{} role to the Workspace Managed Identity. "
                             "Trying to grant Workspace Managed Identity access to the {} "
                             "resource failed with error status '{}' and error message '{}'").format(
                role_name, datastore_type, response.status_code,
                _DatastoreClient._get_error_message(response))
            module_logger.warning(error_message)

    @staticmethod
    @_handle_error_response_exception
    def _check_or_grant_access(subscription_id, resource_group, workspace, account_name, custom_warning,
                               datastore_type):
        rest_client = _DatastoreClient._get_client(workspace, workspace._auth, None)
        url = None

        # Get resource manager endpoint from auth
        resource_manager_endpoint = workspace._auth._cloud_type.endpoints.resource_manager

        if datastore_type == "Blob" or datastore_type == "AzureDataLakeGen2":
            url = urljoin(resource_manager_endpoint,
                          '/subscriptions/{subscription_id}/resourceGroups/{'
                          'resource_group}/providers/Microsoft.Storage/storageAccounts/{'
                          'account_name}/providers/Microsoft.Authorization/roleAssignments')
        elif datastore_type == "AzureDataLake":
            url = urljoin(resource_manager_endpoint,
                          '/subscriptions/{subscription_id}/resourceGroups/{'
                          'resource_group}/providers/Microsoft.DataLakeStore/Accounts/{account_name}/'
                          'providers/Microsoft.Authorization/roleAssignments')
        elif datastore_type == "AzureSqlDatabase":
            url = urljoin(resource_manager_endpoint,
                          '/subscriptions/{subscription_id}/resourceGroups/{'
                          'resource_group}/providers/Microsoft.Sql/Servers/{account_name}/'
                          'providers/Microsoft.Authorization/roleAssignments')
        else:
            raise UserErrorException("Managed Identity access is currently supported for storage accounts Azure Blob, "
                                     "AzureDataLake, AzureDataLakeGen2, AzureSqlDatabase. "
                                     "The access to any other storage is not supported.")

        path_format_arguments = {
            'subscription_id': rest_client._serialize.url("subscription_id", subscription_id, 'str'),
            'resource_group': rest_client._serialize.url("resource_group", resource_group, 'str'),
            'account_name': rest_client._serialize.url("account_name", account_name, 'str')
        }
        url = rest_client._client.format_url(url, **path_format_arguments)

        # Construct parameters
        workspace_resource = _commands.get_workspace(workspace._auth, workspace.subscription_id,
                                                     workspace.resource_group, workspace.name)
        query_parameters = {}
        if workspace_resource.primary_user_assigned_identity:
            module_logger.debug("getting user assigned identity.")
            # check the user assigned identities manually as the case sensitivity is dfferent if it is get from ARM.
            for key, value in workspace_resource.identity.user_assigned_identities.items():
                if key.lower() == workspace_resource.primary_user_assigned_identity.lower():
                    principal_id = value.principal_id
                    break
            if not principal_id:
                raise Exception("The primary user assigned identity : "
                                + workspace_resource.primary_user_assigned_identity
                                + " does not exsit in the workspace.")
        else:
            module_logger.debug("getting system assigned identity.")
            principal_id = workspace_resource.identity.principal_id
        query_parameters['$filter'] = quote("principalId eq '" + principal_id + "'")
        query_parameters['api-version'] = '2015-07-01'

        # Construct headers
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json'
        header_parameters['Authentication'] = 'Bearer ' + workspace._auth._get_arm_token()

        # Construct and send request
        request = rest_client._client.get(url, params=query_parameters)
        response = rest_client._client.send(
            request, header_parameters, None, stream=False)

        # Reader and Storage Blob Data Reader
        reader = ROLE_ID_OF_READER
        storage_blob_data_reader = ROLE_ID_OF_STORAGE_BLOB_DATA_READER
        missing_roles = {reader}
        missing_roles_for_blob = {reader, storage_blob_data_reader}

        if response.status_code == 200:
            json_obj = json.loads(response.text)
            granted_roles = set([role['properties']['roleDefinitionId'].split('/')[-1] for role in json_obj['value']])

            if datastore_type == "Blob" or datastore_type == "AzureDataLakeGen2":
                missing_roles = missing_roles_for_blob - granted_roles
            else:
                missing_roles = missing_roles - granted_roles

            if not missing_roles:
                return
        if response.status_code == 404:
            raise Exception("The storage you specified doesn't exist.")
        else:
            module_logger.warning(custom_warning.format(rest_client._deserialize('ErrorResponse', response)))

        for role in missing_roles:
            module_logger.debug("adding misssing role: " + role)
            _DatastoreClient._grant_workspace_access(workspace, role, url, datastore_type, principal_id)

    @staticmethod
    def register_azure_blob_container(workspace, datastore_name, container_name, account_name, sas_token=None,
                                      account_key=None, protocol=None, endpoint=None, overwrite=False,
                                      create_if_not_exists=False, skip_validation=False, blob_cache_timeout=None,
                                      grant_workspace_access=False, subscription_id=None, resource_group=None):
        """Register an Azure Blob Container to the datastore.

        You can choose to use SAS Token or Storage Account Key

        :param workspace: The workspace.
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: The name of the datastore, case insensitive, can only contain alphanumeric characters
            and _
        :type datastore_name: str
        :param container_name: The name of the azure blob container.
        :type container_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param sas_token: An account SAS token, defaults to None.
        :type sas_token: str, optional
        :param account_key: A storage account key, defaults to None.
        :type account_key: str, optional
        :param protocol: Protocol to use to connect to the blob container. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
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
        :param grant_workspace_access: Grant the Workspace Managed Identity (MSI) access to the user storage account,
            defaults to False. Set it to True to access data behind virtual network from Machine Learning Studio.
            This makes data access from Machine Learning Studio use workspace managed identity for authentication,
            and adds the workspace managed identity as Reader of the storage. You have to be owner or user access
            administrator of the storage to opt-in. Ask your administrator to configure it for you if you do not
            have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        :param subscription_id: The subscription id of the storage account, defaults to None.
        :type subscription_id: str, optional
        :param resource_group: The resource group of the storage account, defaults to None.
        :type resource_group: str, optional
        :return: The blob datastore.
        :rtype: AzureBlobDatastore
        """
        import azureml.data.constants as constants

        if grant_workspace_access:
            if not subscription_id:
                raise UserErrorException("Storage Account's subscription ID is needed in order to grant the Workspace "
                                         "Managed Identity access to the Storage Account.")
            if not resource_group:
                raise UserErrorException("Storage Account's resource group is needed in order to grant the Workspace "
                                         "Managed Identity access to the Storage Account.")
            warning = "You do not have permissions to check whether the Workspace Managed Identity has access to " \
                      "the Storage Account. The check failed with '{}'. We will try to grant Reader and Storage " \
                      "Blob Data Reader role to the Workspace Managed Identity for the storage account."
            _DatastoreClient._check_or_grant_access(subscription_id, resource_group, workspace, account_name, warning,
                                                    "Blob")

        credential_type = _DatastoreClient._get_credential_type(account_key, sas_token)
        return _DatastoreClient._register_azure_storage(
            ws=workspace,
            datastore_name=datastore_name,
            storage_type=constants.AZURE_BLOB,
            container_name=container_name,
            account_name=account_name,
            credential_type=credential_type,
            credential=sas_token or account_key,
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

        :param workspace: The workspace.
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: The name of the datastore, case insensitive, can only contain alphanumeric characters
            and _
        :type datastore_name: str
        :param file_share_name: The name of the azure file container.
        :type file_share_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param sas_token: An account SAS token, defaults to None.
        :type sas_token: str, optional
        :param account_key: A storage account key, defaults to None.
        :type account_key: str, optional
        :param protocol: Protocol to use to connect to the file share. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
            it will create one, defaults to False
        :type overwrite: bool, optional
        :param create_if_not_exists: create the file share if it does not exists, defaults to False
        :type create_if_not_exists: bool, optional
        :param skip_validation: skips validation of storage keys, defaults to False
        :type skip_validation: bool, optional
        :return: The file datastore.
        :rtype: AzureFileDatastore
        """
        import azureml.data.constants as constants

        credential_type = _DatastoreClient._get_credential_type(account_key, sas_token)
        return _DatastoreClient._register_azure_storage(
            workspace, datastore_name, constants.AZURE_FILE, file_share_name, account_name, credential_type,
            sas_token or account_key, protocol, endpoint, overwrite, create_if_not_exists, skip_validation)

    @staticmethod
    def register_azure_data_lake(workspace, datastore_name, store_name, tenant_id, client_id, client_secret,
                                 resource_url=None, authority_url=None, subscription_id=None, resource_group=None,
                                 overwrite=False, grant_workspace_access=False):
        """Initialize a new Azure Data Lake Datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param store_name: the ADLS store name
        :type store_name: str
        :param tenant_id: the Directory ID/Tenant ID of the service principal
        :type tenant_id: str
        :param client_id: the Client ID/Application ID of the service principal
        :type client_id: str
        :param client_secret: the secret of the service principal
        :type client_secret: str
        :param resource_url: the resource url, which determines what operations will be performed on
            the data lake store, defaults to https://datalake.azure.net/ which allows us to perform filesystem
            operations
        :type resource_url: str, optional
        :param authority_url: the authority url used to authenticate the user, defaults to
            https://login.microsoftonline.com
        :type authority_url: str, optional
        :param subscription_id: the ID of the subscription the ADLS store belongs to
        :type subscription_id: str, optional
        :param resource_group: the resource group the ADLS store belongs to
        :type resource_group: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
            it will create one, defaults to False
        :type overwrite: bool, optional
        :param grant_workspace_access: Grant the Workspace Managed Identity (MSI) access to the user storage account,
            defaults to False. Set it to True to access data behind virtual network from Machine Learning Studio.
            This makes data access from Machine Learning Studio use workspace managed identity for authentication,
            and adds the workspace managed identity as Reader of the storage. You have to be owner or user access
            administrator of the storage to opt-in. Ask your administrator to configure it for you if you do not
            have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        """
        if grant_workspace_access:
            if not subscription_id:
                raise UserErrorException("Storage Account's subscription ID is needed in order to grant the Workspace "
                                         "Managed Identity access to the Storage Account.")
            if not resource_group:
                raise UserErrorException("Storage Account's resource group is needed in order to grant the Workspace "
                                         "Managed Identity access to the Storage Account.")
            warning = "You do not have permissions to check whether the Workspace Managed Identity has access to " \
                      "the Storage Account. The check failed with '{}'. You can assign the workspace-managed " \
                      "identity access to resources just like any other security principle. For more information, " \
                      "see Access control in Azure Data Lake Storage Gen1. " \
                      "https://docs.microsoft.com/azure/machine-learning/" \
                      "how-to-enable-studio-virtual-network#azure-sql-database-contained-user"
            _DatastoreClient._check_or_grant_access(subscription_id, resource_group, workspace, store_name, warning,
                                                    "AzureDataLake")

        return _DatastoreClient._register_azure_data_lake(
            workspace, datastore_name, store_name, tenant_id, client_id, client_secret, resource_url,
            authority_url, subscription_id, resource_group, overwrite, grant_workspace_access=grant_workspace_access)

    @staticmethod
    def register_azure_data_lake_gen2(workspace, datastore_name, container_name, account_name, protocol, endpoint,
                                      tenant_id, client_id, client_secret, resource_url=None, authority_url=None,
                                      overwrite=False, subscription_id=None, resource_group=None,
                                      grant_workspace_access=False):
        """Initialize a new Azure Data Lake Gen2 Datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param container_name: The name of the azure blob container.
        :type container_name: str
        :param account_name: The storage account name.
        :type account_name: str
        :param protocol: Protocol to use to connect to the blob container. If None, defaults to https.
        :type protocol: str, optional
        :param endpoint: The endpoint of the blob container. If None, defaults to core.windows.net.
        :type endpoint: str, optional
        :param tenant_id: the Directory ID/Tenant ID of the service principal
        :type tenant_id: str
        :param client_id: the Client ID/Application ID of the service principal
        :type client_id: str
        :param client_secret: the secret of the service principal
        :type client_secret: str
        :param resource_url: the resource url, which determines what operations will be performed on
            the data lake store, defaults to https://storage.azure.com/ which allows us to perform filesystem
            operations
        :type resource_url: str, optional
        :param authority_url: the authority url used to authenticate the user, defaults to
            https://login.microsoftonline.com
        :type authority_url: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
            it will create one, defaults to False
        :type overwrite: bool, optional
        :param subscription_id: the ID of the subscription the ADLS store belongs to
        :type subscription_id: str, optional
        :param resource_group: the resource group the ADLS store belongs to
        :type resource_group: str, optional
        :param grant_workspace_access: Grant the Workspace Managed Identity (MSI) access to the user storage account,
            defaults to False. Set it to True to access data behind virtual network from Machine Learning Studio.
            This makes data access from Machine Learning Studio use workspace managed identity for authentication,
            and adds the workspace managed identity as Reader of the storage. You have to be owner or user access
            administrator of the storage to opt-in. Ask your administrator to configure it for you if you do not
            have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        """
        if grant_workspace_access:
            if not subscription_id:
                raise UserErrorException("Storage Account's subscription ID is needed in order to grant the Workspace "
                                         "Managed Identity access to the Storage Account.")
            if not resource_group:
                raise UserErrorException("Storage Account's resource group is needed in order to grant the Workspace "
                                         "Managed Identity access to the Storage Account.")
            warning = "You do not have permissions to check whether the Workspace Managed Identity has access to " \
                      "the Storage Account. The check failed with '{}'. You can use both RBAC and POSIX-style " \
                      "access control lists (ACLs) to control data access inside of a virtual network." \
                      "To use RBAC, add the workspace-managed identity to the Blob Data Reader role. " \
                      "For more information, see Azure role-based access control." \
                      "To use ACLs, the workspace-managed identity can be assigned access just like any other " \
                      "security principle. For more information, see Access control lists on files and directories."
            _DatastoreClient._check_or_grant_access(subscription_id, resource_group, workspace,
                                                    account_name, warning, "AzureDataLakeGen2")
        return _DatastoreClient._register_azure_data_lake_gen2(
            workspace, datastore_name, container_name, account_name, tenant_id, client_id, protocol, endpoint,
            client_secret, resource_url, authority_url, overwrite, subscription_id=subscription_id,
            resource_group=resource_group, grant_workspace_access=grant_workspace_access)

    @staticmethod
    def register_azure_sql_database(workspace, datastore_name, server_name, database_name, tenant_id, client_id,
                                    client_secret, resource_url=None, authority_url=None, endpoint=None,
                                    overwrite=False, username=None, password=None, skip_validation=False,
                                    subscription_id=None, resource_group=None, grant_workspace_access=False):
        """Initialize a new Azure SQL database Datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param server_name: the SQL server name
        :type server_name: str
        :param database_name: the SQL database name
        :type database_name: str
        :param tenant_id: the Directory ID/Tenant ID of the service principal
        :type tenant_id: str
        :param client_id: the Client ID/Application ID of the service principal
        :type client_id: str
        :param client_secret: the secret of the service principal
        :type client_secret: str
        :param resource_url: the resource url, which determines what operations will be performed on
            the SQL database store, if None, defaults to https://database.windows.net/
        :type resource_url: str, optional
        :param authority_url: the authority url used to authenticate the user, defaults to
        https://login.microsoftonline.com
        :type authority_url: str, optional
        :param endpoint: The endpoint of the SQL server. If None, defaults to database.windows.net.
        :type endpoint: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
            it will create one, defaults to False
        :type overwrite: bool, optional
        :param username: The username of the database user to access the database.
        :type username: str
        :param password: The password of the database user to access the database.
        :type password: str
        :param skip_validation: Whether to skip validation of connecting to the SQL database. Defaults to False.
        :type skip_validation: bool, optional
        :param subscription_id: the ID of the subscription the ADLS store belongs to
        :type subscription_id: str, optional
        :param resource_group: the resource group the ADLS store belongs to
        :type resource_group: str, optional
        :param grant_workspace_access: Grant the Workspace Managed Identity (MSI) access to the user storage account,
            defaults to False. Set it to True to access data behind virtual network from Machine Learning Studio.
            This makes data access from Machine Learning Studio use workspace managed identity for authentication,
            and adds the workspace managed identity as Reader of the storage. You have to be owner or user access
            administrator of the storage to opt-in. Ask your administrator to configure it for you if you do not
            have the required permission.
            Learn more 'https://docs.microsoft.com/azure/machine-learning/how-to-enable-studio-virtual-network'
        :type grant_workspace_access: bool, optional
        """
        if grant_workspace_access:
            if not subscription_id:
                raise UserErrorException("Storage Account's subscription ID is needed in order to grant the Workspace "
                                         "Managed Identity access to the Azure SQL Server.")
            if not resource_group:
                raise UserErrorException("Storage Account's resource group is needed in order to grant the Workspace "
                                         "Managed Identity access to the Azure SQL Server.")
            warning = "You do not have permissions to check whether the Workspace Managed Identity has access to " \
                      "the Azure SQL Server. The check failed with '{}'. To access data stored in an Azure SQL " \
                      "Database using managed identity, you must create a SQL contained user that maps to the " \
                      "managed identity. For more information on creating a user from an external provider, see " \
                      "Create contained users mapped to Azure AD identities. After you create a SQL contained user, " \
                      "grant permissions to it by using the GRANT T-SQL command. " \
                      "https://docs.microsoft.com/sql/t-sql/statements/grant-object-permissions-transact-sql"
            _DatastoreClient._check_or_grant_access(subscription_id, resource_group, workspace,
                                                    server_name, warning, "AzureSqlDatabase")
        return _DatastoreClient._register_azure_sql_database(
            workspace, datastore_name, server_name, database_name, tenant_id, client_id, client_secret,
            resource_url, authority_url, endpoint, overwrite, username=username, password=password,
            skip_validation=skip_validation, subscription_id=subscription_id,
            resource_group=resource_group, grant_workspace_access=grant_workspace_access)

    @staticmethod
    def register_azure_postgre_sql(workspace, datastore_name, server_name, database_name,
                                   user_id, user_password, port_number=None, endpoint=None,
                                   overwrite=False, enforce_ssl=True, skip_validation=False):
        """Initialize a new Azure PostgreSQL Datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param server_name: the PostgreSQL server name
        :type server_name: str
        :param database_name: the PostgreSQL database name
        :type database_name: str
        :param user_id: the User Id of the PostgreSQL server
        :type user_id: str
        :param user_password: the User Password of the PostgreSQL server
        :type user_password: str
        :param port_number: the Port Number of the PostgreSQL server
        :type port_number: str
        :param endpoint: The endpoint of the PostgreSQL server. If None, defaults to postgres.database.azure.com.
        :type endpoint: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
        it will create one, defaults to False
        :type overwrite: bool, optional
        :param enforce_ssl: Indicates SSL requirement of PostgreSQL server. Defaults to True.
        :type enforce_ssl: bool
        :param skip_validation: Whether to skip validation of connecting to the SQL database. Defaults to False.
        :type skip_validation: bool, optional
        """
        return _DatastoreClient._register_azure_postgre_sql(
            workspace, datastore_name, server_name, database_name, user_id, user_password,
            port_number, endpoint, overwrite, enforce_ssl, skip_validation=skip_validation)

    @staticmethod
    def register_azure_my_sql(workspace, datastore_name, server_name, database_name,
                              user_id, user_password, port_number=None, endpoint=None,
                              overwrite=False, skip_validation=False):
        """Initialize a new Azure MySQL Datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param server_name: the MySQL server name
        :type server_name: str
        :param database_name: the MySQL database name
        :type database_name: str
        :param user_id: the User Id of the MySQL server
        :type user_id: str
        :param user_password: the User Password of the MySQL server
        :type user_password: str
        :param port_number: the Port Number of the MySQL server
        :type port_number: str
        :param endpoint: The endpoint of the MySQL server. If None, defaults to mysql.database.azure.com.
        :type endpoint: str, optional
        :param overwrite: overwrites an existing datastore. If the datastore does not exist,
        it will create one, defaults to False
        :type overwrite: bool, optional
        :param skip_validation: Whether to skip validation of connecting to the SQL database. Defaults to False.
        :type skip_validation: bool, optional
        """
        return _DatastoreClient._register_azure_my_sql(
            workspace, datastore_name, server_name, database_name, user_id, user_password,
            port_number, endpoint, overwrite, skip_validation=skip_validation)

    @staticmethod
    def register_dbfs(workspace, datastore_name):
        """Initialize a new Databricks File System (DBFS) datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param name: the datastore name
        :type name: str
        """
        return _DatastoreClient._register_dbfs(workspace, datastore_name, overwrite=True)

    @staticmethod
    def register_hdfs(workspace, datastore_name, protocol, namenode_address, hdfs_server_certificate,
                      kerberos_realm, kerberos_kdc_address, kerberos_principal, kerberos_keytab=None,
                      kerberos_password=None, overwrite=False):
        """Initialize a new HDFS datastore.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: the datastore name
        :type datastore_name: str
        :param protocol: The protocol to use when communicating with the HDFS cluster.
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
        return _DatastoreClient._register_hdfs(workspace, datastore_name, protocol, namenode_address,
                                               hdfs_server_certificate, kerberos_realm, kerberos_kdc_address,
                                               kerberos_principal, kerberos_keytab, kerberos_password, overwrite)

    @staticmethod
    def list(workspace, count=None):
        """List all of the datastores in the workspace. List operation does not return credentials of the datastores.

        :param workspace: the workspace this datastore belongs to
        :type workspace: azureml.core.workspace.Workspace
        :param count: the number of datastores to retrieve. If None or 0, will retrieve all datastores. This might
            take some time depending on the number of datastores.
        :type count: int
        :return: List of datastores
        :rtype: list[AzureFileDatastore] or list[AzureBlobDatastore] or list[AzureDataLakeDatastore]
        or list[AzureSqlDatabaseDatastore] or list[AzurePostgreSqlDatastore]
        """
        datastores = []
        ct = None

        if not count:
            while True:
                dss, ct = _DatastoreClient._list(workspace, ct, 100)
                datastores += dss

                if not ct:
                    break
        else:
            dss, ct = _DatastoreClient._list(workspace, None, count)
            datastores += dss

        return datastores

    @staticmethod
    def delete(workspace, datastore_name):
        """Delete a datastore.

        :param workspace: The workspace
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: The datastore name to delete
        :type datastore_name: str
        """
        _DatastoreClient._delete(workspace, datastore_name)

    @staticmethod
    def get_default(workspace):
        """Get the default datastore for the workspace.

        :param workspace: The workspace
        :type workspace: azureml.core.workspace.Workspace
        :return: The default datastore for the workspace
        :rtype: typing.Union[AzureFileDatastore, AzureBlobDatastore]
        """
        return _DatastoreClient._get_default(workspace)

    @staticmethod
    def set_default(workspace, datastore_name):
        """Set the default datastore for the workspace.

        :param workspace: The workspace
        :type workspace: azureml.core.workspace.Workspace
        :param datastore_name: The name of the datastore to be set as the default
        :type datastore_name: str
        """
        _DatastoreClient._set_default(workspace, datastore_name)

    @staticmethod
    @_handle_error_response_exception
    def _get(ws, name, auth=None, host=None):
        module_logger.debug("Getting datastore: {}".format(name))

        client = _DatastoreClient._get_client(ws, auth, host)
        datastore = client.data_stores.get(subscription_id=ws._subscription_id, resource_group_name=ws._resource_group,
                                           workspace_name=ws._workspace_name, name=name,
                                           custom_headers=_DatastoreClient._custom_headers)

        module_logger.debug("Received DTO from the datastore service")
        return _DatastoreClient._dto_to_datastore(ws, datastore)

    @staticmethod
    def _register_azure_storage(ws, datastore_name, storage_type, container_name, account_name,
                                credential_type, credential, protocol, endpoint, overwrite,
                                create_if_not_exists, skip_validation, auth=None, host=None,
                                blob_cache_timeout=None, grant_workspace_access=False, subscription_id=None,
                                resource_group=None):
        module_logger.debug("Registering {} datastore".format(storage_type))
        service_data_access_auth_identity = _DatastoreClient._get_service_data_access_auth_identity(
            ws, grant_workspace_access)
        storage_dto = AzureStorage(account_name=account_name, container_name=container_name, endpoint=endpoint,
                                   protocol=protocol, credential_type=credential_type, credential=credential,
                                   blob_cache_timeout=blob_cache_timeout, subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   are_workspace_managed_identities_allowed=grant_workspace_access,
                                   service_data_access_auth_identity=service_data_access_auth_identity)
        datastore = DataStore(name=datastore_name, data_store_type=storage_type, azure_storage_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=create_if_not_exists,
                                          skip_validation=skip_validation, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_azure_data_lake(ws, datastore_name, store_name, tenant_id, client_id, client_secret,
                                  resource_url=None, authority_url=None, subscription_id=None, resource_group=None,
                                  overwrite=False, auth=None, host=None, grant_workspace_access=False):
        import azureml.data.constants as constants

        module_logger.debug("Registering {} datastore".format(constants.AZURE_DATA_LAKE))
        resource_url = resource_url or constants.ADLS_RESOURCE_URI
        service_data_access_auth_identity = _DatastoreClient._get_service_data_access_auth_identity(
            ws, grant_workspace_access)
        storage_dto = AzureDataLake(store_name=store_name, authority_url=authority_url, resource_uri=resource_url,
                                    tenant_id=tenant_id, client_id=client_id, is_cert_auth=False,
                                    client_secret=client_secret, subscription_id=subscription_id,
                                    resource_group=resource_group,
                                    service_data_access_auth_identity=service_data_access_auth_identity)
        datastore = DataStore(name=datastore_name, data_store_type=constants.AZURE_DATA_LAKE,
                              azure_data_lake_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False,
                                          skip_validation=False, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_azure_data_lake_gen2(ws, datastore_name, container_name, account_name, tenant_id, client_id,
                                       protocol, endpoint, client_secret, resource_url=None, authority_url=None,
                                       overwrite=False, auth=None, host=None, subscription_id=None,
                                       resource_group=None, grant_workspace_access=False):
        import azureml.data.constants as constants

        module_logger.debug("Registering {} datastore".format(constants.AZURE_DATA_LAKE_GEN2))

        cred_type = constants.NONE
        if tenant_id and client_secret and client_id:
            cred_type = constants.CLIENT_CREDENTIALS
        elif tenant_id or client_secret or client_id:
            raise UserErrorException("Tenant id, client id and client secret have to be defined together.")

        resource_url = resource_url or constants.STORAGE_RESOURCE_URI
        cred = ClientCredentials(client_id=client_id, tenant_id=tenant_id, is_cert_auth=False,
                                 client_secret=client_secret, resource_uri=resource_url,
                                 authority_url=authority_url)
        service_data_access_auth_identity = _DatastoreClient._get_service_data_access_auth_identity(
            ws, grant_workspace_access)
        storage_dto = AzureStorage(account_name=account_name, container_name=container_name, endpoint=endpoint,
                                   protocol=protocol, credential_type=cred_type, client_credentials=cred,
                                   subscription_id=subscription_id, resource_group=resource_group,
                                   service_data_access_auth_identity=service_data_access_auth_identity)
        datastore = DataStore(name=datastore_name, data_store_type=constants.AZURE_DATA_LAKE_GEN2,
                              azure_storage_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False,
                                          skip_validation=False, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_azure_sql_database(ws, datastore_name, server_name, database_name, tenant_id=None, client_id=None,
                                     client_secret=None, resource_url=None, authority_url=None, endpoint=None,
                                     overwrite=False, auth=None, host=None, username=None, password=None,
                                     skip_validation=False, subscription_id=None, resource_group=None,
                                     grant_workspace_access=False):
        import azureml.data.constants as constants

        module_logger.debug("Registering {} datastore".format(constants.AZURE_SQL_DATABASE))
        service_data_access_auth_identity = _DatastoreClient._get_service_data_access_auth_identity(
            ws, grant_workspace_access)
        auth_type = constants.NONE
        if username and password:
            auth_type = constants.SQL_AUTHENTICATION
        elif tenant_id and client_id and client_secret:
            auth_type = constants.SERVICE_PRINCIPAL

        storage_dto = AzureSqlDatabase(server_name=server_name, database_name=database_name,
                                       authority_url=authority_url, resource_uri=resource_url,
                                       tenant_id=tenant_id, client_id=client_id, is_cert_auth=False,
                                       client_secret=client_secret, endpoint=endpoint, user_id=username,
                                       user_password=password, credential_type=auth_type,
                                       subscription_id=subscription_id, resource_group=resource_group,
                                       service_data_access_auth_identity=service_data_access_auth_identity)
        datastore = DataStore(name=datastore_name, data_store_type=constants.AZURE_SQL_DATABASE,
                              azure_sql_database_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False,
                                          skip_validation=skip_validation, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_azure_postgre_sql(ws, datastore_name, server_name, database_name, user_id, user_password,
                                    port_number=None, endpoint=None, overwrite=False, enforce_ssl=True, auth=None,
                                    host=None, skip_validation=False):
        import azureml.data.constants as constants

        module_logger.debug("Registering {} datastore".format(constants.AZURE_POSTGRESQL))
        storage_dto = AzurePostgreSql(server_name=server_name, database_name=database_name,
                                      user_id=user_id, user_password=user_password,
                                      port_number=port_number, enable_ssl=enforce_ssl, endpoint=endpoint)
        datastore = DataStore(name=datastore_name, data_store_type=constants.AZURE_POSTGRESQL,
                              azure_postgre_sql_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False,
                                          skip_validation=skip_validation, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_azure_my_sql(ws, datastore_name, server_name, database_name, user_id, user_password,
                               port_number=None, endpoint=None, overwrite=False, auth=None, host=None,
                               skip_validation=False):
        import azureml.data.constants as constants

        module_logger.debug("Registering {} datastore".format(constants.AZURE_MYSQL))
        storage_dto = AzureMySql(server_name=server_name, database_name=database_name,
                                 user_id=user_id, user_password=user_password,
                                 port_number=port_number, endpoint=endpoint)
        datastore = DataStore(name=datastore_name, data_store_type=constants.AZURE_MYSQL,
                              azure_my_sql_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False,
                                          skip_validation=skip_validation, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_dbfs(ws, datastore_name, overwrite=False, auth=None, host=None):
        import azureml.data.constants as constants

        module_logger.debug("Registering {} datastore".format(constants.DBFS))
        datastore = DataStore(name=datastore_name, data_store_type=constants.DBFS)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False, skip_validation=False,
                                          overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    def _register_hdfs(ws, datastore_name, protocol, namenode_address, hdfs_server_certificate,
                       kerberos_realm, kerberos_kdc_address, kerberos_principal, kerberos_keytab=None,
                       kerberos_password=None, overwrite=False, auth=None, host=None):
        import azureml.data.constants as constants
        from base64 import b64encode

        module_logger.debug(f"Registering {constants.HDFS} datastore")

        if kerberos_password and kerberos_keytab:
            raise UserErrorException("Found both a keytab and a password. Specify one or the other, not both.")

        if kerberos_password:
            credential_value = kerberos_password
            credential_type = HdfsCredentialType.kerberos_password
        elif kerberos_keytab:
            with open(kerberos_keytab, 'rb') as f:
                credential_value = b64encode(f.read()).decode('utf-8')
                credential_type = HdfsCredentialType.kerberos_keytab
        else:
            raise UserErrorException("Either a password or a keytab must be specified.")

        if hdfs_server_certificate:
            with open(hdfs_server_certificate, 'rb') as f:
                cert_data = b64encode(f.read()).decode('utf-8')
        else:
            cert_data = None

        storage_dto = OnPremHdfs(protocol=protocol, namenode_address=namenode_address,
                                 hdfs_server_certificate=cert_data, kerberos_realm=kerberos_realm,
                                 kerberos_kdc_address=kerberos_kdc_address, kerberos_principal=kerberos_principal,
                                 credential_value=credential_value, credential_type=credential_type)
        datastore = DataStore(name=datastore_name, data_store_type=constants.HDFS, hdfs_section=storage_dto)
        module_logger.debug("Converted data into DTO")
        return _DatastoreClient._register(ws=ws, dto=datastore, create_if_not_exists=False,
                                          skip_validation=True, overwrite=overwrite, auth=auth, host=host)

    @staticmethod
    @_handle_error_response_exception
    def _register(ws, dto, create_if_not_exists, skip_validation, overwrite, auth, host):
        try:
            if _DatastoreClient._capital_regex.search(dto.name):
                module_logger.warning("Datastore name {} contains capital letters. ".format(dto.name)
                                      + "They will be converted to lowercase letters.")
            client = _DatastoreClient._get_client(ws, auth, host)
            module_logger.debug("Posting DTO to datastore service")
            client.data_stores.create(ws._subscription_id, ws._resource_group, ws._workspace_name,
                                      dto, create_if_not_exists, skip_validation,
                                      custom_headers=_DatastoreClient._custom_headers)
        except HttpOperationError as e:
            if e.response.status_code == 400 and overwrite and CONFLICT_MESSAGE in e.message:
                module_logger.info("Failed to create datastore because another datastore with the same name already "
                                   "exists. Trying to update existing datastore's credential now.")
                client = _DatastoreClient._get_client(ws, auth, host)
                client.data_stores.update(subscription_id=ws._subscription_id, resource_group_name=ws._resource_group,
                                          workspace_name=ws._workspace_name, name=dto.name, dto=dto,
                                          create_if_not_exists=create_if_not_exists, skip_validation=skip_validation,
                                          custom_headers=_DatastoreClient._custom_headers)
            else:
                module_logger.error("Registering datastore failed with a {} error code and error message '{}'"
                                    .format(e.response.status_code, _DatastoreClient._get_http_exception_message(e)))
                raise e

        return _DatastoreClient._get(ws, dto.name, auth, host)

    @staticmethod
    @_handle_error_response_exception
    def _list(ws, continuation_token, count, auth=None, host=None):
        module_logger.debug("Listing datastores with continuation token: {}".format(continuation_token or ""))
        client = _DatastoreClient._get_client(ws, auth, host)
        datastore_dtos = client.data_stores.list(ws._subscription_id, ws._resource_group, ws._workspace_name,
                                                 continuation_token=continuation_token, count=count,
                                                 custom_headers=_DatastoreClient._custom_headers)
        datastores = filter(lambda ds: ds is not None,
                            map(lambda dto: _DatastoreClient._dto_to_datastore(ws, dto), datastore_dtos.value))
        return list(datastores), datastore_dtos.continuation_token

    @staticmethod
    @_handle_error_response_exception
    def _delete(ws, name, auth=None, host=None):
        module_logger.debug("Deleting datastore: {}".format(name))
        client = _DatastoreClient._get_client(ws, auth, host)
        client.data_stores.delete(subscription_id=ws._subscription_id, resource_group_name=ws._resource_group,
                                  workspace_name=ws._workspace_name, name=name,
                                  custom_headers=_DatastoreClient._custom_headers)

    @staticmethod
    @_handle_error_response_exception
    def _get_default(ws, auth=None, host=None):
        module_logger.debug("Getting default datastore for provided workspace")
        client = _DatastoreClient._get_client(ws, auth, host)
        ds = client.data_stores.get_default(ws._subscription_id, ws._resource_group, ws._workspace_name,
                                            custom_headers=_DatastoreClient._custom_headers)
        return _DatastoreClient._dto_to_datastore(ws, ds)

    @staticmethod
    @_handle_error_response_exception
    def _set_default(ws, name, auth=None, host=None):
        module_logger.debug("Setting default datastore for provided workspace to: {}".format(name))
        client = _DatastoreClient._get_client(ws, auth, host)
        client.data_stores.set_default(subscription_id=ws._subscription_id, resource_group_name=ws._resource_group,
                                       workspace_name=ws._workspace_name, name=name,
                                       custom_headers=_DatastoreClient._custom_headers)

    @staticmethod
    def _get_client(ws, auth, host):
        host_env = os.environ.get('AZUREML_SERVICE_ENDPOINT')
        auth = auth or ws._auth
        host = host or host_env or get_service_url(
            auth, _DatastoreClient._get_workspace_uri_path(ws._subscription_id, ws._resource_group,
                                                           ws._workspace_name), ws._workspace_id, ws.discovery_url)

        return RestClient(credentials=_DatastoreClient._get_basic_token_auth(auth), base_url=host)

    @staticmethod
    def _get_basic_token_auth(auth):
        return BasicTokenAuthentication({
            "access_token": _DatastoreClient._get_access_token(auth)
        })

    @staticmethod
    def _get_access_token(auth):
        module_logger.info(auth)
        header = auth.get_authentication_header()
        bearer_token = header["Authorization"]

        return bearer_token[_DatastoreClient._bearer_prefix_len:]

    @staticmethod
    def _get_workspace_uri_path(subscription_id, resource_group, workspace_name):
        return ("/subscriptions/{}/resourceGroups/{}/providers"
                "/Microsoft.MachineLearningServices"
                "/workspaces/{}").format(subscription_id, resource_group, workspace_name)

    @staticmethod
    def _dto_to_datastore(ws, datastore):
        import azureml.data.constants as constants

        datastore_type = datastore.data_store_type.value

        def empty_value_to_none(v):
            return None if v == '00000000-0000-0000-0000-000000000000' or v == '' else v

        if datastore_type == constants.AZURE_BLOB:
            as_section = datastore.azure_storage_section
            service_data_access_auth_identity = constants.NONE
            if "serviceDataAccessAuthIdentity" in as_section.additional_properties:
                service_data_access_auth_identity = as_section.additional_properties["serviceDataAccessAuthIdentity"]
            return AzureBlobDatastore(
                ws, datastore.name, as_section.container_name, as_section.account_name,
                as_section.sas_token, as_section.account_key, as_section.protocol, as_section.endpoint,
                subscription_id=as_section.subscription_id, resource_group=as_section.resource_group,
                service_data_access_auth_identity=service_data_access_auth_identity)
        if datastore_type == constants.AZURE_FILE:
            as_section = datastore.azure_storage_section
            return AzureFileDatastore(
                ws, datastore.name, as_section.container_name, as_section.account_name,
                as_section.sas_token, as_section.account_key, as_section.protocol, as_section.endpoint)
        if datastore_type == constants.AZURE_DATA_LAKE:
            ad_section = datastore.azure_data_lake_section
            service_data_access_auth_identity = constants.NONE
            if "serviceDataAccessAuthIdentity" in ad_section.additional_properties:
                service_data_access_auth_identity = ad_section.additional_properties["serviceDataAccessAuthIdentity"]
            return AzureDataLakeDatastore(
                ws, datastore.name, ad_section.store_name, empty_value_to_none(ad_section.tenant_id),
                empty_value_to_none(ad_section.client_id), empty_value_to_none(ad_section.client_secret),
                ad_section.resource_uri, ad_section.authority_url,
                ad_section.subscription_id, ad_section.resource_group,
                service_data_access_auth_identity=service_data_access_auth_identity)
        if datastore_type == constants.AZURE_SQL_DATABASE:
            ad_section = datastore.azure_sql_database_section
            service_data_access_auth_identity = constants.NONE
            if "serviceDataAccessAuthIdentity" in ad_section.additional_properties:
                service_data_access_auth_identity = ad_section.additional_properties["serviceDataAccessAuthIdentity"]
            return AzureSqlDatabaseDatastore(
                ws, datastore.name, ad_section.server_name, ad_section.database_name,
                ad_section.tenant_id, ad_section.client_id, ad_section.client_secret,
                ad_section.resource_uri, ad_section.authority_url, ad_section.user_id,
                ad_section.user_password,
                service_data_access_auth_identity=service_data_access_auth_identity)
        if datastore_type == constants.AZURE_POSTGRESQL:
            psql_section = datastore.azure_postgre_sql_section
            return AzurePostgreSqlDatastore(
                ws, datastore.name, psql_section.server_name, psql_section.database_name,
                psql_section.user_id, psql_section.user_password, psql_section.port_number, psql_section.enable_ssl)
        if datastore_type == constants.AZURE_MYSQL:
            mysql_section = datastore.azure_my_sql_section
            return AzureMySqlDatastore(
                ws, datastore.name, mysql_section.server_name, mysql_section.database_name,
                mysql_section.user_id, mysql_section.user_password, mysql_section.port_number)
        if datastore_type == constants.AZURE_DATA_LAKE_GEN2:
            adlg2_section = datastore.azure_storage_section
            cred_section = adlg2_section.client_credentials
            service_data_access_auth_identity = constants.NONE
            if "serviceDataAccessAuthIdentity" in adlg2_section.additional_properties:
                service_data_access_auth_identity = \
                    adlg2_section.additional_properties["serviceDataAccessAuthIdentity"]
            gen2_ds = AzureDataLakeGen2Datastore(
                ws, datastore.name, adlg2_section.container_name, adlg2_section.account_name,
                protocol=adlg2_section.protocol, endpoint=adlg2_section.endpoint,
                service_data_access_auth_identity=service_data_access_auth_identity)
            if cred_section:
                gen2_ds.tenant_id = cred_section.tenant_id
                gen2_ds.client_id = cred_section.client_id
                gen2_ds.client_secret = cred_section.client_secret
                gen2_ds.resource_url = cred_section.resource_uri
                gen2_ds.authority_url = cred_section.authority_url
            return gen2_ds
        if datastore_type == constants.DBFS:
            return DBFSDatastore(ws, datastore.name)
        if datastore_type == constants.HDFS:
            hdfs_section = datastore.hdfs_section
            hdfs_datastore = HDFSDatastore(ws, datastore.name, hdfs_section.protocol,
                                           hdfs_section.namenode_address, hdfs_section.hdfs_server_certificate,
                                           hdfs_section.kerberos_realm, hdfs_section.kerberos_kdc_address,
                                           hdfs_section.kerberos_principal, hdfs_section.credential_value,
                                           hdfs_section.credential_type)

            return hdfs_datastore
        if datastore_type == constants.CUSTOM:
            # TODO return AbstractDatastore for custom datastore, which is now only available to 1P
            # consider using new class CustomDatastore if we make it public
            return AbstractDatastore(ws, datastore.name, constants.CUSTOM)
        raise TypeError("Unsupported Datastore Type: {}".format(datastore.data_store_type))

    @staticmethod
    def _get_credential_type(account_key, sas_token):
        import azureml.data.constants as constants

        if account_key:
            return constants.ACCOUNT_KEY
        if sas_token:
            return constants.SAS
        return constants.NONE

    @staticmethod
    def _get_http_exception_message(ex):
        try:
            error = json.loads(ex.response.text)
            return error["error"]["message"]
        except Exception:
            return ex.message

    @staticmethod
    def _get_error_message(response):
        try:
            error = json.loads(response.text)
            return error["error"]["message"]
        except Exception:
            return response

    @staticmethod
    def _get_error_code(response):
        try:
            error = json.loads(response.text)
            return error["error"]["code"]
        except Exception:
            return response

    @staticmethod
    def _get_service_data_access_auth_identity(workspace, grant_workspace_access):
        if grant_workspace_access:
            workspace_resource = _commands.get_workspace(workspace._auth, workspace.subscription_id,
                                                         workspace.resource_group, workspace.name)
            if workspace_resource.primary_user_assigned_identity:
                return ServiceDataAccessAuthIdentity.workspace_user_assigned_identity
            else:
                return ServiceDataAccessAuthIdentity.workspace_system_assigned_identity
        else:
            return ServiceDataAccessAuthIdentity.none
