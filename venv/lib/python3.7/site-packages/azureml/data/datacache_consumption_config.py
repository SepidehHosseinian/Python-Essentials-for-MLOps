# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for DataCache consumption configuration."""
import random
import string
from azureml.data.constants import MOUNT_MODE
from azureml.exceptions import UserErrorException
from azureml.data.datacache_client import _DatacacheClient
from azureml._base_sdk_common._docstring_wrapper import experimental


@experimental
class DatacacheConsumptionConfig:
    """Represent how to deliver the datacache to the compute target.

    :param datacache_store: The datacachestore to be used to cache the dataset.
    :type datacache_store: azureml.data.datacache.DatacacheStore
    :param dataset: The dataset that needs to be cached.
    :type dataset: azureml.data.file_dataset.FileDataset
    :param mode: Defines how the datacache should be delivered to the compute target. Only mount mode
        is supported currently.
    :type mode: str
    :param replica_count: Number of replicas to be used by the job.
    :type replica_count: Int, Optional, Defaults to 1
    :param path_on_compute: The target path on the compute to make the data available at.
    :type path_on_compute: str, Optional
    """

    _SUPPORTED_MODE = {MOUNT_MODE}

    def __init__(self, datacache_store, dataset, mode=MOUNT_MODE,
                 replica_count=None, path_on_compute=None, **kwargs):
        """Represent how to deliver the datacache to the compute target.

        :param datacache_store: The datacachestore to be used to cache the dataset.
        :type datacache_store: azureml.data.datacache.DatacacheStore
        :param dataset: The dataset that needs to be cached.
        :type dataset: azureml.data.file_dataset.FileDataset
        :param mode: Defines how the datacache should be delivered to the compute target. Only mount mode
            is supported currently.
        :type mode: str
        :param replica_count: Number of replicas to be used by the training job.
        :type replica_count: Int, Optional
        :param path_on_compute: The target path on the compute to make the data available at.
        :type path_on_compute: str, Optional
        """
        self.datacache_store = datacache_store
        self.dataset = dataset
        mode = mode.lower()
        DatacacheConsumptionConfig._validate_mode(mode)
        self.mode = mode
        self.replica_count = replica_count
        self.path_on_compute = path_on_compute
        self.datacache_id = kwargs.get("_datacache_id", None)

        if self.datacache_id is None:
            dc = _DatacacheClient.get_datacache(self.datacache_store.workspace, self.datacache_store, self.dataset)
            self.datacache_id = dc._id

    def as_mount(self, path_on_compute=None, replica_count=None):
        """Set the datacache mode to mount from requested number of replicas.

        In the submitted run, required replicas will be mounted to local path on the compute target.
        The local path on the compute can be retrieved from the arguments in the run.

        :param path_on_compute: The target path on the compute to make the data available at.
        :type path_on_compute: str, Optional
        :param replica_count: Number of replicas
        :type replica_count: Int
        """
        if path_on_compute is None:
            path_on_compute = '/tmp/tmp' + ''.join(random.choice(string.ascii_lowercase) for i in range(8))

        dto = _DatacacheClient.consume_cache(self.datacache_store.workspace, self.datacache_id, replica_count)
        if not dto:
            raise UserErrorException("None of the replicas are available for use now. Please consider hydrating"
                                     + " the cache first.")

        self.path_on_compute = path_on_compute
        self.replica_count = replica_count

        return self

    @staticmethod
    def _validate_mode(mode):
        if mode not in DatacacheConsumptionConfig._SUPPORTED_MODE:
            raise UserErrorException("Invalid mode '{}'. Mode can only be mount".format(mode))
