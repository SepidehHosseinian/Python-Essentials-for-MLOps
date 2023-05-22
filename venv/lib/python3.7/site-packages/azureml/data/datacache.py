# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing DatacacheStore and Datacache in Azure Machine Learning."""
import logging
import os
import time
import sys

from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml.exceptions import UserErrorException, AzureMLException
from azureml.core.datastore import Datastore
from azureml.core.run import Run
from collections import OrderedDict

module_logger = logging.getLogger(__name__)


@experimental
class DatacacheStore(object):
    """Represents a storage abstraction over an Azure Machine Learning storage account.

    DatacacheStores are attached to workspaces and are used to store information related to the underlying datacache
    solution. Currently, only partitioned blob solution is supported. Datacachestores defines various Blob datastores
    that could be used for caching.

    Use this class to perform management operations, including register, list, get, and update datacachestores.
    DatacacheStores for each service are created with the ``register*`` methods of this class.
    """

    # Constructor will be used as a getter
    def __init__(self, workspace, name, **kwargs):
        """Get a datacachestore by name. This call will make a request to the datacache service.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datacachestore
        :type name: str
        :return: The corresponding datastore for that name.
        :rtype: DatacacheStore
        """
        self._workspace = None
        self._name = None
        self._id = None
        self._data_store_list = None
        self._data_management_compute_target = None
        self._data_factory_resource_id = None
        self._ttl_in_days = None
        self._ttl_expiration_policy = None
        self._default_replica_count = None
        self._created_by = None

        _datacache_store_dto = kwargs.get("_datacache_store_dto", None)
        if _datacache_store_dto:
            self._initialize(workspace, datacache_store_dto=_datacache_store_dto)
        elif workspace and name:
            dto = DatacacheStore._client()._get_datacache_store(workspace, name)
            self._initialize(workspace, dto)
        else:
            raise UserErrorException("DatacacheStore name is required.")

    def __repr__(self):
        """Return the string representation of the DatacacheStore object.

        :return: String representation of the DatacacheStore object
        :rtype: str
        """
        datastore_names = []
        for datastores in self.data_store_list:
            datastore_names.append(datastores.name)

        info = OrderedDict([
            ('name', self.name),
            ('workspace', self.workspace.name if self.workspace is not None else None),
            ('data_store_list', datastore_names),
            ('data_management_compute_target', self._data_management_compute_target),
            ('data_factory_resource_id', self._data_factory_resource_id),
            ('ttl_in_days', self._ttl_in_days),
            ('ttl_expiration_policy', self._ttl_expiration_policy),
            ('default_replica_count', self._default_replica_count),
            ('created_by', self._created_by)
        ])

        formatted_info = ','.join(["{}: {}".format(k, v) for k, v in info.items()])
        return "DatacacheStore({0})".format(formatted_info)

    def __str__(self):
        """Return the string representation of the DatacacheStore object.

        :return: String representation of the DatacacheStore object
        :rtype: str
        """
        return self.__repr__()

    def _initialize(self, workspace, datacache_store_dto):
        self._workspace = workspace
        self._name = datacache_store_dto.name
        self._id = datacache_store_dto.id
        datastore_list = []
        for ds in datacache_store_dto.datastore_names:
            datastore_list.append(Datastore.get(workspace, ds))
        self._data_store_list = datastore_list
        if datacache_store_dto.cache_policy is not None:
            self._data_management_compute_target = datacache_store_dto.cache_policy.data_management_compute_name
            if datacache_store_dto.cache_policy.data_hydration_external_resources is not None:
                adf = datacache_store_dto.cache_policy.data_hydration_external_resources.azure_data_factory
                self._data_factory_resource_id = (
                    '/subscriptions/{}/resourceGroups/{}/providers/Microsoft.DataFactory/factories/{}'
                    .format(adf.subscription_id, adf.resource_group_name, adf.name))

        if datacache_store_dto.cache_policy is not None:
            self._default_replica_count = datacache_store_dto.cache_policy.default_replica_number

        if datacache_store_dto.cache_policy is not None and datacache_store_dto.cache_policy.ttl is not None:
            self._ttl_in_days = datacache_store_dto.cache_policy.ttl.ttl_in_day
            self._ttl_expiration_policy = datacache_store_dto.cache_policy.ttl.based_on

        if datacache_store_dto.created_by is not None:
            self._created_by = datacache_store_dto.created_by.user_name

    @property
    def name(self):
        """Return the name of the datacache store.

        :return: The DatacacheStore name.
        :rtype: str
        """
        return self._name

    @property
    def workspace(self):
        """Return the workspace information.

        :return: The workspace.
        :rtype: azureml.core.Workspace
        """
        return self._workspace

    @property
    def data_store_list(self):
        """Return the list of underlying datastores for the datacachestores.

        :return: The list of datastores to be used as datacachestores.
        :rtype: builtin.list(AbstractDataStore)
        """
        return self._data_store_list

    @property
    def data_management_compute_target(self):
        """Return the name of the data management compute to be used for hydration.

        :return: The data management compute.
        :rtype: str
        """
        return self._data_management_compute_target

    @property
    def data_factory_resource_id(self):
        """Return the resource id of the Azure data factory which can be used for hydration.

        :return: Resource id of the ADF to be used for hydration.
        :rtype: str
        """
        return self._data_factory_resource_id

    @property
    def ttl_in_days(self):
        """Return the Time-to-live policy.

        :return: Time-To-Live in days.
        :rtype: Int
        """
        return self._ttl_in_days

    @property
    def ttl_expiration_policy(self):
        """Return the Time-to-live expiration policy.

        :return: TTL expire policy.
        :rtype: str
        """
        return self._ttl_expiration_policy

    @property
    def default_replica_count(self):
        """Return the default number of replicas during hydration.

        :return: Default number of replicas to hydrate.
        :rtype: Int
        """
        return self._default_replica_count

    @staticmethod
    def register(workspace, name, data_store_list, data_management_compute_target,
                 data_management_compute_auth, ttl_in_days, ttl_expiration_policy,
                 default_replica_count,
                 data_factory_resource_id=None,
                 **kwargs):
        """Register a datacachestore to the workspace.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datacachestore.
        :type name: str
        :param data_store_list: The list of underlying datastores.
        :type name: typing.Union[builtin.list(str), builtin.list(AbstractDataStore)]
        :param data_management_compute_target: The data management compute.
        :type data_management_compute_target: azureml.core.compute.ComputeTarget
        :param data_management_compute_auth: The service principal used to submit data management jobs
            to data management compute.
        :type data_management_compute_auth:  azureml.core.authentication.ServicePrincipalAuthentication
        :param ttl_in_days:  Time-To-Live in days.
        :type ttl_in_days: Int
        :param ttl_expiration_policy: TTL expire policy.
        :type ttl_expiration_policy: str, one of ["LastAccessTime", "CreationTime"]
        :param default_replica_count:  Default number of replicas to hydrate.
        :type default_replica_count: Int
        :param data_factory_resource_id: Resource id of the ADF to be used for hydration.
        :type data_factory_resource_id: str.
        :return: The DatacacheStore object
        :rtype: azureml.data.datacache.DatacacheStore
        """
        singularity_settings = kwargs.get("singularity_settings")

        return DatacacheStore._client().register_datacache_store(
            workspace, name, data_store_list, data_management_compute_target,
            data_management_compute_auth, ttl_in_days, ttl_expiration_policy,
            default_replica_count, data_factory_resource_id,
            singularity_settings=singularity_settings)

    @staticmethod
    def update(workspace, name, data_management_compute_target=None, data_management_compute_auth=None,
               ttl_in_days=None, ttl_expiration_policy=None,
               default_replica_count=None, data_factory_resource_id=None,
               **kwargs):
        """Update datacache policy for a datacachestore.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datacachestore.
        :type name: str
        :param data_management_compute_target: The data management compute.
        :type data_management_compute_target: azureml.core.compute.ComputeTarget
        :param data_management_compute_auth: The service principal used to submit data management jobs
            to data management compute.
        :type data_management_compute_auth:  azureml.core.authentication.ServicePrincipalAuthentication
        :param ttl_in_days:  Time-To-Live in days.
        :type ttl_in_days: Int
        :param ttl_expiration_policy: TTL expire policy.
        :type ttl_expiration_policy: str, one of ["LastAccessTime", "CreationTime"]
        :param default_replica_count:  Default number of replicas to hydrate.
        :type default_replica_count: Int
        :param data_factory_resource_id: Resource id of the ADF to be used for hydration.
        :type data_factory_resource_id: str.
        :return: The DatacacheStore object
        :rtype: azureml.data.datacache.DatacacheStore
        """
        singularity_settings = kwargs.get("singularity_settings")

        return DatacacheStore._client().update_datacache_store_policy(workspace, name, data_management_compute_target,
                                                                      data_management_compute_auth,
                                                                      ttl_in_days, ttl_expiration_policy,
                                                                      default_replica_count, data_factory_resource_id,
                                                                      singularity_settings=singularity_settings)

    @staticmethod
    def get_by_name(workspace, name):
        """Get a datacachestore by name.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datacachestore
        :type name: str
        :return: The corresponding datastore for that name.
        :rtype: azureml.data.datacache.DatacacheStore
        """
        return DatacacheStore._client().get_datacache_store(workspace, name)

    @staticmethod
    def list(workspace):
        """List all the datacachestores in the workspace.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :return: List of DatacacheStore objects.
        :rtype: builtin.list[azureml.data.datacache.DatacacheStore]
        """
        return DatacacheStore._client().list_datacache_stores(workspace)

    @staticmethod
    def _client():
        """Get a client.

        :return: Returns the client
        :rtype: DatacacheClient
        """
        from azureml.data.datacache_client import _DatacacheClient
        return _DatacacheClient


@experimental
class DatacacheHydrationTracker(object):
    """DatacacheHydrationTracker is used to keep track of the hydration job.

    It can be used to wait on the the completion of the hydration process before consuming the datacache
    in the training job.
    """

    _WAIT_COMPLETION_POLLING_INTERVAL_MIN = os.environ.get("AZUREML_RUN_POLLING_INTERVAL_MIN", 20)
    _WAIT_COMPLETION_POLLING_INTERVAL_MAX = os.environ.get("AZUREML_RUN_POLLING_INTERVAL_MAX", 60)

    def __init__(self, workspace, datacache_store, dataset, replica_count, run_id):
        """Create the datacache hydration tracker object.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: The name of the datacachestore.
        :type name: str
        :param dataset: The dataset that is being hydrated.
        :type dataset: azureml.data.file_dataset.FileDataset
        :param replica_count: Number of replicas requested for the hydration job.
        :type replica_count: Int
        :param run_id: Run id of the hydration job to track.
        :type run_id: str
        :return: Datacache Hydration Tracker.
        :rtype: DatacacheHydrationTracker
        """
        self._workspace = workspace
        self._datacache_store = datacache_store
        self._dataset = dataset
        self._replica = replica_count
        self._run_id = run_id
        if replica_count is None:
            self._replica = datacache_store.default_replica_count
        self._datacache = _Datacache.get(workspace, datacache_store, dataset)

    @staticmethod
    def _wait_before_polling(current_seconds):
        if current_seconds < 0:
            raise ValueError("current_seconds must be positive")
        import math
        # Sigmoid that tapers off near the max at ~ 3 min
        duration = (DatacacheHydrationTracker._WAIT_COMPLETION_POLLING_INTERVAL_MAX
                    / (1.0 + 100 * math.exp(-current_seconds / 20.0)))

        return max(DatacacheHydrationTracker._WAIT_COMPLETION_POLLING_INTERVAL_MIN, duration)

    def _stream_run_output(self, file_handle=sys.stdout):

        file_handle.write("DatacacheId: {}\n".format(self._datacache._id))
        available_replicas = 0
        if self._run_id:
            run = Run.get(self._workspace, self._run_id)
            file_handle.write("Hydration Run Portal URL: {}\n".format(run.get_portal_url()))
            poll_start_time = time.time()
            file_handle.write("Waiting for the hydration run to start... \n")
            while run.get_status() in ['Queued', 'NotStarted', 'Preparing', 'Provisioning', 'Starting']:
                time.sleep(DatacacheHydrationTracker._wait_before_polling(time.time() - poll_start_time))

        poll_start_time = time.time()
        while available_replicas < self._replica:
            file_handle.flush()
            time.sleep(DatacacheHydrationTracker._wait_before_polling(time.time() - poll_start_time))
            datacache_dto = _Datacache._client()._get_datacache(self._workspace, self._datacache_store.name,
                                                                self._dataset.id)
            (ready_to_use, pending_hydration, hydrating) = self._parse_dto(datacache_dto)

            available_replicas = ready_to_use

            file_handle.write("Requested Replicas = " + str(self._replica) + " ; " + "Available Replicas = "
                              + str(available_replicas) + " ; ")
            file_handle.write("Replicas hydrating = " + str(hydrating) + " ; " + "Replicas waiting for hydration = "
                              + str(pending_hydration) + "\n")
            file_handle.write("=================\n")

            if (available_replicas + pending_hydration + hydrating) < self._replica \
                    and (time.time() - poll_start_time) > 1200:
                module_logger.error("Number of available replicas " + str(available_replicas)
                                    + " Number of replicas pending hydration " + str(pending_hydration)
                                    + " Number of replicas hydrating " + str(hydrating)
                                    + " Number of requested replicas " + str(self._replica))
                raise AzureMLException("The expected number of replicas are not ready for consumption. "
                                       + "The hydration job seems to have failed due to some reason. Please consider"
                                       + "running hydrate again for this datacache")

        file_handle.write("\n")
        file_handle.flush()

    def _parse_dto(self, datacache_dto):
        ready_to_use = 0
        pending_hydration = 0
        hydrating = 0

        for replica in datacache_dto.replicas:
            if replica.status.value == "HydratingSuccess" or replica.status.value == "InRead":
                ready_to_use += 1

            if replica.status.value == "WaitingHydrate":
                pending_hydration += 1

            if replica.status.value == "Hydrating":
                hydrating += 1

        return (ready_to_use, pending_hydration, hydrating)

    def wait_for_completion(self, show_output=False):
        """Wait for the requested replicas to be hydrated.

        :param show_output: Indicates whether to provide more verbose output.
        :type show_output: bool, Defaults to False
        :return: The status object.
        :rtype: dict
        """
        if show_output:
            try:
                self._stream_run_output(
                    file_handle=sys.stdout)
            except KeyboardInterrupt:
                error_message = "The output streaming for the datacache hydration tracker is interrupted.\n" \
                                "But the hydration is still executing on the backend. \n"
                raise AzureMLException(error_message)
        else:
            available_replicas = 0
            if self._run_id:
                run = Run.get(self._workspace, self._run_id)
                poll_start_time = time.time()
                while run.get_status() in ['Queued', 'NotStarted', 'Preparing', 'Provisioning', 'Starting']:
                    time.sleep(DatacacheHydrationTracker._wait_before_polling(time.time() - poll_start_time))

            poll_start_time = time.time()
            while available_replicas < self._replica:
                time.sleep(DatacacheHydrationTracker._wait_before_polling(time.time() - poll_start_time))
                datacache_dto = _Datacache._client()._get_datacache(self._workspace,
                                                                    self._datacache_store.name,
                                                                    self._dataset.id)
                (ready_to_use, pending_hydration, hydrating) = self._parse_dto(datacache_dto)

                available_replicas = ready_to_use

                if (available_replicas + pending_hydration + hydrating) < self._replica \
                        and (time.time() - poll_start_time) > 1200:
                    module_logger.error("Number of available replicas " + str(available_replicas)
                                        + " Number of pending hydration " + str(pending_hydration)
                                        + " Number of requested replicas " + str(self._replica))
                    raise AzureMLException("The expected number of replicas are not ready for consumption. "
                                           + "The hydration job seems to have failed due to some reason. "
                                           + "Please consider running hydrate again for this datacache")


class _Datacache(object):
    """Represents a data cache instance for a given DatacacheStore and a Dataset.

    Datacache are created to serve the data required by the training jobs. They are created to cache the underlying
    training data near the compute in order to improve GPU/ CPU utilization and lower latency of
    the training jobs. Datacache supports various actions to cache the corresponding Datasets into
    DatacacheStores.

    Use this class to perform actions on the Datacache, including create, get and hydrate datacaches.
    """

    def __init__(self, workspace, datacache_store, dataset, **kwargs):
        from azureml.data.file_dataset import FileDataset

        self._workspace = None
        self._datacache_store = None
        self._dataset = None
        self._id = None
        self._path_on_datacache_store = None
        self._num_replicas_available_for_consumption = None
        self._num_unavailable_replicas = None
        self._num_replicas_hydrating = None
        self._num_replicas_hydrated = None
        self._num_replicas_pending_hydration = None

        _datacache_dto = kwargs.get("_datacache_dto", None)
        if _datacache_dto:
            self._initialize(workspace, datacache_dto=_datacache_dto)
        elif workspace and datacache_store and dataset:
            if not isinstance(datacache_store, DatacacheStore):
                raise TypeError("datacache_store should be of type azureml.core.DatacacheStore")

            if not isinstance(dataset, FileDataset):
                raise TypeError("dataset should be of type azureml.data.FileDataset")

            if not dataset.id:
                dataset._ensure_saved(workspace)

            dto = _Datacache._client()._get_datacache(workspace, datacache_store.name, dataset.id)
            self._initialize(workspace, datacache_dto=dto)
        else:
            raise UserErrorException("Datacache definition isn't correct.")

    def __repr__(self):

        info = OrderedDict()
        info['id'] = self._id
        info['dataset'] = self.dataset.__repr__()
        info['datacache_store'] = self.datacache_store.__repr__()
        info['workspace'] = self._workspace.name
        info['path_on_datacache_store'] = self.path_on_datacache_store
        info['num_replicas_hydrated'] = self._num_replicas_hydrated
        info['num_replicas_hydrating'] = self._num_replicas_hydrating
        # info['num_replicas_pending_hydration'] = self._num_replicas_pending_hydration

        formatted_info = ',\n'.join(["{}: {}".format(k, v) for k, v in info.items()])
        return "Datacache(\n{0})".format(formatted_info)

    def __str__(self):
        return self.__repr__()

    def _initialize(self, workspace, datacache_dto):
        from azureml.core.dataset import Dataset

        self._workspace = workspace
        self._id = datacache_dto.id
        self._datacache_store = DatacacheStore.get_by_name(workspace, datacache_dto.datacache_store_name)
        self._dataset = Dataset.get_by_id(workspace, datacache_dto.saved_dataset_id)
        self._path_on_datacache_store = datacache_dto.path_on_datacache_store
        self._registered_time = datacache_dto.registered_time
        self._num_replicas_available_for_consumption = 0
        self._num_replicas_hydrated = 0
        self._num_replicas_hydrating = 0
        self._num_replicas_pending_hydration = 0
        self._num_unavailable_replicas = 0

        for replica in datacache_dto.replicas:
            if replica.status.value == "HydratingSuccess" or replica.status.value == "InRead":
                self._num_replicas_available_for_consumption += 1
                self._num_replicas_hydrated += 1
            elif replica.status.value == "Hydrating":
                self._num_replicas_hydrating += 1
            elif replica.status.value == "WaitingHydrate":
                self._num_replicas_pending_hydration += 1
            else:
                self._num_unavailable_replicas += 1

    @property
    def workspace(self):
        return self._workspace

    @property
    def datacache_store(self):
        return self._datacache_store

    @property
    def dataset(self):

        return self._dataset

    @property
    def path_on_datacache_store(self):
        return self._path_on_datacache_store

    @property
    def num_replicas_available_for_consumption(self):
        return self._num_replicas_available_for_consumption

    @property
    def num_unavailable_replicas(self):
        return self._num_unavailable_replicas

    @staticmethod
    def get(workspace, datacache_store, dataset):
        return _Datacache._client().get_datacache(workspace, datacache_store, dataset)

    @staticmethod
    def create(workspace, datacache_store, dataset):
        return _Datacache._client().create_datacache(workspace, datacache_store, dataset)

    def hydrate(self, replica=None):

        if replica is not None and (replica < 1 or not isinstance(replica, int)):
            raise UserErrorException("replica should be an integer greater than 0")

        dto = _Datacache._client().hydrate_datacache(self.workspace, self._id, replica)
        return DatacacheHydrationTracker(self.workspace, self.datacache_store, self.dataset, replica, dto.run_id)

    def delete(self):
        return _Datacache._client()._delete(self.workspace, self._id)

    @staticmethod
    def _client():
        """Get a client.

        :return: Returns the client
        :rtype: DatacacheClient
        """
        from azureml.data.datacache_client import _DatacacheClient
        return _DatacacheClient
