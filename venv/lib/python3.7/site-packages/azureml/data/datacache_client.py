# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Internal use only."""
import logging
import os
import json

from azureml.exceptions import UserErrorException
from msrest.authentication import BasicTokenAuthentication
from msrest.exceptions import HttpOperationError
from azureml._base_sdk_common.service_discovery import get_service_url
from azureml._base_sdk_common import _ClientSessionId
from azureml._restclient.rest_client import RestClient
from azureml._restclient.models.create_datacache_store_request import CreateDatacacheStoreRequest
from azureml._restclient.models.datacache_store_policy_input import DatacacheStorePolicyInput
from azureml._restclient.models.data_management_compute_definition import DataManagementComputeDefinition
from azureml._restclient.models.singularity_settings import SingularitySettings
from azureml._restclient.models.azure_data_factory import AzureDataFactory
from azureml._restclient.models.data_hydration_external_resources import DataHydrationExternalResources
from azureml._restclient.models.service_principal_definition import ServicePrincipalDefinition
from azureml._restclient.models.create_datacache_request import CreateDatacacheRequest
from azureml._restclient.models.datacache_hydration_request import DatacacheHydrationRequest
from azureml._restclient.models.time_to_live import TimeToLive
from azureml._restclient.models.update_cache_policy_request import UpdateCachePolicyRequest
from azureml._restclient.models.update_consumption_run_request import UpdateConsumptionRunRequest
from azureml.core.compute import ComputeTarget
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.file_dataset import FileDataset
from azureml.core.authentication import ServicePrincipalAuthentication

module_logger = logging.getLogger(__name__)


class _DatacacheClient:
    """A client that provides methods to communicate with the datacache service."""

    # the auth token received from _auth.get_authentication_header is prefixed
    # with 'Bearer '. This is used to remove that prefix.
    _bearer_prefix_len = 7

    # the unique id of each python kernel process on client side to
    # correlate events within each process.
    _custom_headers = {"x-ms-client-session-id": _ClientSessionId}

    @staticmethod
    def register_datacache_store(workspace, name, data_store_list, data_management_compute_target,
                                 data_management_compute_auth, ttl_in_days, ttl_expiration_policy,
                                 default_replica_count, data_factory_resource_id,
                                 singularity_settings=None):

        from azureml.data.datacache import DatacacheStore

        # check if all the elements in the list are either of type string or AbstractDatastore
        if not data_store_list or type(data_store_list) != list:
            raise UserErrorException("data_store_list should be an non-empty list.")

        if not isinstance(name, str) or name == "":
            raise UserErrorException("name should be a non-empty str.")

        if singularity_settings:  # Singularity job
            if not isinstance(singularity_settings.instance_type, str) or singularity_settings.instance_type == "":
                raise UserErrorException("singularity_settings.instance_type should be a non-empty str.")
            if not isinstance(singularity_settings.virtual_cluster_arm_id, str) or \
                    singularity_settings.virtual_cluster_arm_id == "":
                raise UserErrorException("singularity_settings.virtual_cluster_arm_id should be a non-empty str.")
        else:  # AmlCompute/ITP job
            if (not isinstance(data_management_compute_target, ComputeTarget)
                    or data_management_compute_target.type not in ["AmlCompute", "Cmk8s"]):
                raise UserErrorException("data_management_compute_target should be of instance ComputeTarget and only \
                                        AmlCompute/ITP cluster are supported targets.")

        if not isinstance(data_management_compute_auth, ServicePrincipalAuthentication):
            raise UserErrorException("data_management_compute_auth should be of type ServicePrincipalAuthentication")

        if not ttl_in_days or type(ttl_in_days) != int:
            raise UserErrorException("ttl_in_days should be an integer greater than 0")

        if ttl_expiration_policy != 'LastAccessTime' and ttl_expiration_policy != 'CreationTime':
            raise UserErrorException("ttl_expiration_policy should be either `LastAccessTime` or `CreationTime`")

        if default_replica_count > len(data_store_list) or default_replica_count < 1 \
                or not isinstance(default_replica_count, int):
            raise UserErrorException("default_replica_count should be lesser than or equal to the number of"
                                     + "datastores provided in data_store_list"
                                     + "and an integer greater than 0")

        # TODO: make this more robust by perform check via MLC query
        if data_factory_resource_id:
            # If ADF is provided, validate ADF resource id
            resource_parts = data_factory_resource_id.split('/')
            if len(resource_parts) != 9:
                raise UserErrorException('Invalid resource_id provided: {}'.format(data_factory_resource_id))
            resource_type = resource_parts[6]
            if resource_type != 'Microsoft.DataFactory':
                raise UserErrorException('Invalid resource_id provided, resource type {} does not match for '
                                         'DataFactory'.format(resource_type))

        dto = None
        if all(isinstance(x, AzureBlobDatastore) for x in data_store_list):
            ds_names = []
            for ds in data_store_list:
                ds_names.append(ds.name)
            dto = _DatacacheClient._register_datacache_store(workspace, name, ds_names,
                                                             data_management_compute_target,
                                                             data_management_compute_auth,
                                                             ttl_in_days, ttl_expiration_policy,
                                                             default_replica_count, data_factory_resource_id,
                                                             singularity_settings=singularity_settings)
        elif all(isinstance(x, str) for x in data_store_list):
            dto = _DatacacheClient._register_datacache_store(workspace, name, data_store_list,
                                                             data_management_compute_target,
                                                             data_management_compute_auth,
                                                             ttl_in_days, ttl_expiration_policy,
                                                             default_replica_count, data_factory_resource_id,
                                                             singularity_settings=singularity_settings)
        else:
            raise UserErrorException("All the elements in data_store_list should be of the same type "
                                     "It can be either of type str or type builtin.list(AzureBlobDatastore)")

        return DatacacheStore(workspace, None, _datacache_store_dto=dto)

    @staticmethod
    def update_datacache_store_policy(workspace, name, data_management_compute_target, data_management_compute_auth,
                                      ttl_in_days, ttl_expiration_policy, default_replica_count,
                                      data_factory_resource_id,
                                      singularity_settings=None):

        if not isinstance(name, str) or name == "":
            raise UserErrorException("name should be a non-empty str.")

        if (data_management_compute_target is None and data_factory_resource_id is None) \
                ^ (data_management_compute_auth is None):
            raise UserErrorException("data_management_compute_target/data_factory_resource_id and "
                                     + "data_management_compute_auth needs to be specified together."
                                     + "They cannot be specified individually.")

        if singularity_settings:  # Singularity job
            if not isinstance(singularity_settings.instance_type, str) or singularity_settings.instance_type == "":
                raise UserErrorException("singularity_settings.instance_type should be a non-empty str.")
            if not isinstance(singularity_settings.virtual_cluster_arm_id, str) or \
                    singularity_settings.virtual_cluster_arm_id == "":
                raise UserErrorException("singularity_settings.virtual_cluster_arm_id should be a non-empty str.")
        else:  # AmlCompute/ITP job
            if data_management_compute_target:
                if not isinstance(data_management_compute_target, ComputeTarget) \
                        or data_management_compute_target.type not in ["AmlCompute", "Cmk8s"]:
                    raise UserErrorException("data_management_compute_target should be of instance ComputeTarget and \
                        only AmlCompute/ITP Cluster are supported currently.")

        if (ttl_in_days is None) ^ (ttl_expiration_policy is None):
            raise UserErrorException("ttl_in_days and ttl_expiration_policy needs to be specified together. "
                                     + "They cannot be specified individually.")

        if data_management_compute_auth is not None and \
                not isinstance(data_management_compute_auth, ServicePrincipalAuthentication):
            raise UserErrorException("auth should be of type ServicePrincipalAuthentication")

        if ttl_in_days is not None and (ttl_in_days < 1 or type(ttl_in_days) != int):
            raise UserErrorException("ttl_in_days should be an integer greater than 0")

        if ttl_expiration_policy and (ttl_expiration_policy != 'LastAccessTime'
                                      and ttl_expiration_policy != 'CreationTime'):
            raise UserErrorException("ttl_expiration_policy should be either `LastAccessTime` or `CreationTime`")

        if data_management_compute_target is None and data_management_compute_auth is None \
                and ttl_in_days is None and ttl_expiration_policy is None and default_replica_count is None \
                and data_factory_resource_id is None:
            raise UserErrorException("At least one property should be specified for the update call")

        if default_replica_count is not None and (default_replica_count < 1
                                                  or not isinstance(default_replica_count, int)):
            raise UserErrorException("default_replica_count should be an integer greater than 0")

        # TODO: make this more robust by perform check via MLC query
        if data_factory_resource_id:
            # If ADF is provided, validate ADF resource id
            resource_parts = data_factory_resource_id.split('/')
            if len(resource_parts) != 9:
                raise UserErrorException('Invalid resource_id provided: {}'.format(data_factory_resource_id))
            resource_type = resource_parts[6]
            if resource_type != 'Microsoft.DataFactory':
                raise UserErrorException('Invalid resource_id provided, resource type {} does not match for '
                                         'DataFactory'.format(resource_type))

        dto = _DatacacheClient._update_datacache_store_policy(workspace, name, data_management_compute_target,
                                                              data_management_compute_auth,
                                                              ttl_in_days, ttl_expiration_policy,
                                                              default_replica_count,
                                                              data_factory_resource_id,
                                                              singularity_settings=singularity_settings)

        from azureml.data.datacache import DatacacheStore
        return DatacacheStore(workspace, None, _datacache_store_dto=dto)

    @staticmethod
    def get_datacache_store(workspace, name):
        if not isinstance(name, str) or name == "":
            raise UserErrorException("name should be a non-empty str.")

        dto = _DatacacheClient._get_datacache_store(workspace, name)

        from azureml.data.datacache import DatacacheStore
        return DatacacheStore(workspace, None, _datacache_store_dto=dto)

    @staticmethod
    def list_datacache_stores(workspace):
        datacache_stores = []
        ct = None

        while True:
            dcs_list, ct = _DatacacheClient._list_datacache_stores(workspace, ct)
            datacache_stores += dcs_list

            if not ct:
                break

        return datacache_stores

    @staticmethod
    def create_datacache(workspace, datacache_store, dataset):
        from azureml.data.datacache import _Datacache, DatacacheStore

        if not isinstance(datacache_store, DatacacheStore):
            raise TypeError("datacache_store should be of type azureml.data.DatacacheStore")

        if not isinstance(dataset, FileDataset):
            raise TypeError("dataset should be of type azureml.data.FileDataset")

        if len(json.loads(dataset.__repr__())['source']) != 1:
            raise UserErrorException("Datasets with only single datapath are supoorted for Datacache currently.")

        if not dataset.id:
            dataset._ensure_saved(workspace)

        dto = _DatacacheClient._create_datacache(workspace, datacache_store.name, dataset.id)

        return _Datacache(workspace, None, None, _datacache_dto=dto)

    @staticmethod
    def get_datacache(workspace, datacache_store, dataset):

        from azureml.data.datacache import _Datacache, DatacacheStore
        if not isinstance(datacache_store, DatacacheStore):
            raise TypeError("datacache_store should be of type azureml.data.datacache.DatacacheStore")

        if not isinstance(dataset, FileDataset):
            raise TypeError("dataset should be of type azureml.data.FileDataset")

        if not dataset.id:
            dataset._ensure_saved(workspace)

        dto = _DatacacheClient._get_datacache(workspace, datacache_store.name, dataset.id)

        return _Datacache(workspace, datacache_store, dataset, _datacache_dto=dto)

    @staticmethod
    def hydrate_datacache(workspace, datacache_id, replica):

        dto = _DatacacheClient._hydrate_datacache(workspace, datacache_id, replica)
        return dto

    @staticmethod
    def consume_cache(workspace, datacache_id, replica=None):
        ds_names = _DatacacheClient._consume_cache(workspace, datacache_id, replica)
        return ds_names

    @staticmethod
    def _register_datacache_store(ws, name, data_store_names, data_management_compute_target,
                                  data_management_compute_auth,
                                  ttl_in_days, ttl_expiration_policy,
                                  default_replica_count, data_factory_resource_id, auth=None, host=None,
                                  singularity_settings=None):

        ttl = TimeToLive(ttl_in_day=ttl_in_days, based_on=ttl_expiration_policy)
        sp_def = ServicePrincipalDefinition(client_id=data_management_compute_auth._service_principal_id,
                                            client_secret=data_management_compute_auth._service_principal_password,
                                            tenant_id=data_management_compute_auth._tenant_id)

        data_management_compute_target_name = None
        singularity_settings_def = None
        if singularity_settings:
            singularity_settings_def = SingularitySettings(
                sla_tier=singularity_settings.sla_tier,
                priority=singularity_settings.priority,
                instance_type=singularity_settings.instance_type,
                virtual_cluster_arm_id=singularity_settings.virtual_cluster_arm_id,
                locations=singularity_settings.locations)
        else:
            data_management_compute_target_name = data_management_compute_target.name

        data_management_compute_definition = DataManagementComputeDefinition(
            compute_name=data_management_compute_target_name,
            service_principal_definition=sp_def,
            singularity_settings=singularity_settings_def)

        data_hydration_external_resources = None
        if data_factory_resource_id:
            adf_resource = data_factory_resource_id.split('/')
            azure_data_factory = AzureDataFactory(name=adf_resource[8], subscription_id=adf_resource[2],
                                                  resource_group_name=adf_resource[4],
                                                  service_principal_definition=sp_def)
            data_hydration_external_resources = DataHydrationExternalResources(azure_data_factory=azure_data_factory)

        datacache_store_policy = DatacacheStorePolicyInput(
            ttl=ttl, default_replica_number=default_replica_count,
            data_management_compute_definition=data_management_compute_definition,
            data_hydration_external_resources=data_hydration_external_resources)

        request_body = CreateDatacacheStoreRequest(datacache_store_name=name, datastore_names=data_store_names,
                                                   datacache_store_policy=datacache_store_policy)
        client = _DatacacheClient._get_client(ws, auth, host)
        dto = client.datacache_stores.create(ws._subscription_id, ws._resource_group,
                                             ws._workspace_name, body=request_body,
                                             custom_headers=_DatacacheClient._custom_headers)
        return dto

    @staticmethod
    def _update_datacache_store_policy(ws, name, data_management_compute_target, data_management_compute_auth,
                                       ttl_in_days, ttl_expiration_policy,
                                       default_replica_count, data_factory_resource_id, auth=None, host=None,
                                       singularity_settings=None):
        if ttl_in_days:
            ttl = TimeToLive(ttl_in_day=ttl_in_days, based_on=ttl_expiration_policy)
        else:
            ttl = None

        sp_def = None
        data_management_compute_definition = None
        data_hydration_external_resources = None
        if data_management_compute_auth:
            sp_def = ServicePrincipalDefinition(client_id=data_management_compute_auth._service_principal_id,
                                                client_secret=data_management_compute_auth._service_principal_password,
                                                tenant_id=data_management_compute_auth._tenant_id)

            data_management_compute_target_name = None
            singularity_settings_def = None
            if singularity_settings:
                singularity_settings_def = SingularitySettings(
                    sla_tier=singularity_settings.sla_tier,
                    priority=singularity_settings.priority,
                    instance_type=singularity_settings.instance_type,
                    virtual_cluster_arm_id=singularity_settings.virtual_cluster_arm_id,
                    locations=singularity_settings.locations)
            else:
                data_management_compute_target_name = data_management_compute_target.name

            if data_management_compute_target or singularity_settings:
                data_management_compute_definition = DataManagementComputeDefinition(
                    compute_name=data_management_compute_target_name,
                    service_principal_definition=sp_def,
                    singularity_settings=singularity_settings_def)

            if data_factory_resource_id:
                adf_resource = data_factory_resource_id.split('/')
                azure_data_factory = AzureDataFactory(name=adf_resource[8], subscription_id=adf_resource[2],
                                                      resource_group_name=adf_resource[4],
                                                      service_principal_definition=sp_def)
                data_hydration_external_resources = DataHydrationExternalResources(
                    azure_data_factory=azure_data_factory)

        request_body = UpdateCachePolicyRequest(ttl=ttl, default_replica_number=default_replica_count,
                                                data_management_compute_definition=data_management_compute_definition,
                                                data_hydration_external_resources=data_hydration_external_resources)
        client = _DatacacheClient._get_client(ws, auth, host)
        dto = client.datacache_stores.policy(ws._subscription_id, ws._resource_group,
                                             ws._workspace_name, name, body=request_body,
                                             custom_headers=_DatacacheClient._custom_headers)
        return dto

    @staticmethod
    def _get_datacache_store(ws, name, auth=None, host=None):
        module_logger.debug("Getting datacachestore: {}".format(name))
        client = _DatacacheClient._get_client(ws, auth, host)
        dto = client.datacache_stores.get_by_name(ws._subscription_id, ws._resource_group,
                                                  ws._workspace_name, name, get_policy=True,
                                                  custom_headers=_DatacacheClient._custom_headers)
        module_logger.debug("Received DTO from the datacache service")
        return dto

    @staticmethod
    def _list_datacache_stores(ws, continuation_token, auth=None, host=None):
        from azureml.data.datacache import DatacacheStore
        module_logger.debug("Getting list of datacache_store for the given workspace")
        client = _DatacacheClient._get_client(ws, auth, host)

        dcs_list = client.datacache_stores.list(ws._subscription_id, ws._resource_group,
                                                ws._workspace_name, continuation_token=continuation_token,
                                                get_policy=True, custom_headers=_DatacacheClient._custom_headers)

        dcs = filter(lambda dcs: dcs is not None,
                     map(lambda dto: DatacacheStore(ws, None, _datacache_store_dto=dto), dcs_list.value))

        module_logger.debug("Received DTO from the datacache service")

        return list(dcs), dcs_list.continuation_token

    @staticmethod
    def _get_datacache(ws, datacache_store_name, saved_dataset_id, auth=None, host=None):
        module_logger.debug("Getting datacache instance")
        client = _DatacacheClient._get_client(ws, auth, host)
        datacache_dto = client.datacache.get_by_saved_dataset_id(
            ws._subscription_id, ws._resource_group,
            ws._workspace_name, datacache_store_name, saved_dataset_id,
            custom_headers=_DatacacheClient._custom_headers)
        return datacache_dto

    @staticmethod
    def _hydrate_datacache(ws, datacache_id, replica, auth=None, host=None):
        module_logger.debug("Hydrating datacache instance")
        client = _DatacacheClient._get_client(ws, auth, host)

        request_body = DatacacheHydrationRequest(datacache_id=datacache_id, replica_count=replica)

        hydrate_run = client.datacache.hydration_post(ws._subscription_id, ws._resource_group,
                                                      ws._workspace_name, body=request_body,
                                                      custom_headers=_DatacacheClient._custom_headers)

        return hydrate_run

    @staticmethod
    def _create_datacache(ws, datacache_store_name, saved_dataset_id, auth=None, host=None):
        module_logger.debug("Creating datacache instance")
        client = _DatacacheClient._get_client(ws, auth, host)

        request_body = CreateDatacacheRequest(saved_dataset_id=saved_dataset_id,
                                              datacache_store_name=datacache_store_name)
        datacache_dto = client.datacache.create(ws._subscription_id, ws._resource_group,
                                                ws._workspace_name, body=request_body,
                                                custom_headers=_DatacacheClient._custom_headers)
        return datacache_dto

    @staticmethod
    def _consume_cache(ws, datacache_id, replica, auth=None, host=None):
        module_logger.debug("Calling consume_cache api to get datastore names")
        client = _DatacacheClient._get_client(ws, auth, host)
        dto = client.datacache.get_cache_for_training(ws._subscription_id, ws._resource_group,
                                                      ws._workspace_name, datacache_id, replica,
                                                      custom_headers=_DatacacheClient._custom_headers)
        return dto

    @staticmethod
    def _delete(ws, datacache_id, auth=None, host=None):
        module_logger.debug("Deleting the datacache instance")
        client = _DatacacheClient._get_client(ws, auth, host)
        return client.datacache.delete(ws._subscription_id, ws._resource_group,
                                       ws._workspace_name, datacache_id,
                                       custom_headers=_DatacacheClient._custom_headers)

    @staticmethod
    def _consumption_run_id_update(ws, datacache_id, replica_ids, run_id, auth=None, host=None):
        module_logger.debug("Updating the cache_replica_status")
        client = _DatacacheClient._get_client(ws, auth, host)

        request_body = UpdateConsumptionRunRequest(cache_id=datacache_id, cache_replica_ids=replica_ids,
                                                   run_id=run_id)
        return client.datacache.consumption_run_id_update(ws._subscription_id, ws._resource_group,
                                                          ws._workspace_name, body=request_body,
                                                          custom_headers=_DatacacheClient._custom_headers)

    @staticmethod
    def get_registry_credentials(ws, registry, auth=None, host=None):
        module_logger.debug("Deleting the datacache instance")
        client = _DatacacheClient._get_client(ws, auth, host)
        try:
            cred = client.datacache.registry_cred(ws._subscription_id, ws._resource_group,
                                                  ws._workspace_name, registry,
                                                  custom_headers=_DatacacheClient._custom_headers)
            return cred
        except HttpOperationError as e:
            if e.response.status_code == 404:
                raise e

    @staticmethod
    def _get_client(ws, auth, host):
        host_env = os.environ.get('AZUREML_SERVICE_ENDPOINT')
        auth = auth or ws._auth
        host = host or host_env or get_service_url(
            auth, _DatacacheClient._get_workspace_uri_path(ws._subscription_id, ws._resource_group,
                                                           ws._workspace_name), ws._workspace_id, ws.discovery_url)

        return RestClient(credentials=_DatacacheClient._get_basic_token_auth(auth), base_url=host)

    @staticmethod
    def _get_basic_token_auth(auth):
        return BasicTokenAuthentication({
            "access_token": _DatacacheClient._get_access_token(auth)
        })

    @staticmethod
    def _get_access_token(auth):
        module_logger.info(auth)
        header = auth.get_authentication_header()
        bearer_token = header["Authorization"]

        return bearer_token[_DatacacheClient._bearer_prefix_len:]

    @staticmethod
    def _get_workspace_uri_path(subscription_id, resource_group, workspace_name):
        return ("/subscriptions/{}/resourceGroups/{}/providers"
                "/Microsoft.MachineLearningServices"
                "/workspaces/{}").format(subscription_id, resource_group, workspace_name)

    @staticmethod
    def _get_http_exception_message(ex):
        try:
            error = json.loads(ex.response.text)
            return error["error"]["message"]
        except Exception as e:
            return e.message

    @staticmethod
    def _get_error_message(response):
        try:
            error = json.loads(response.text)
            return error["error"]["message"]
        except Exception as e:
            return e.response

    @staticmethod
    def _get_error_code(response):
        try:
            error = json.loads(response.text)
            return error["error"]["code"]
        except Exception as e:
            return e.response
