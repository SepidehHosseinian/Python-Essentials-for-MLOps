# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains the abstract base class for datasets in Azure Machine Learning."""

import collections
import itertools
import re
import json
import pprint
from abc import ABCMeta
from copy import deepcopy

from azureml.data.dataset_factory import TabularDatasetFactory, FileDatasetFactory
from azureml.data.constants import _TELEMETRY_ENTRY_POINT_DATASET,\
    _PATITION_KEY_ACTIVITY, _PATITION_KEY_VALUES_ACTIVITY
from azureml.data._dataprep_helper import dataprep, is_dataprep_installed, get_dataprep_missing_message
from azureml.data._dataset_rest_helper import _dto_to_dataset, _dataset_to_dto, _dataset_to_saved_dataset_dto, \
    _saved_dataset_dto_to_dataset, _restclient, _dto_to_registration, _custom_headers, _make_request
from azureml.data._loggerfactory import track, _LoggerFactory, collect_datasets_usage
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data._dataset_deprecation import warn_deprecated_blocks
from azureml.exceptions import UserErrorException, RunEnvironmentException


_PUBLIC_API = 'PublicApi'
_INTERNAL_API = 'InternalCall'
_logger = None
_dataprep_missing_for_repr_warned = False
_PATH_COLUMN_NAME = 'Path'
_COMMON_PATH_SUMMARY_COLUMN_NAME = 'azureml_common_path'


def _get_logger():
    global _logger
    if _logger is None:
        _logger = _LoggerFactory.get_logger(__name__)
    return _logger


class AbstractDataset(object):
    """Base class of datasets in Azure Machine Learning.

    Please reference :class:`azureml.data.dataset_factory.TabularDatasetFactory` class and
    :class:`azureml.data.dataset_factory.FileDatasetFactory` class to create instances of dataset.
    """

    __metaclass__ = ABCMeta

    Tabular = TabularDatasetFactory
    File = FileDatasetFactory

    def __init__(self):
        """Class AbstractDataset constructor.

        This constructor is not supposed to be invoked directly. Dataset is intended to be created using
        :class:`azureml.data.dataset_factory.TabularDatasetFactory` class and
        :class:`azureml.data.dataset_factory.FileDatasetFactory` class.
        """
        if self.__class__ == AbstractDataset:
            raise UserErrorException('Cannot create instance of abstract class AbstractDataset')
        self._definition = None
        self._properties = None
        self._registration = None
        self._telemetry_info = None

        # Execution/Pipelines services enable this flag so the latest version of the Dataset is used on the data plane
        self._consume_latest = False

    def __getitem__(self, key):
        """Dataset column indexer.

        :param key: Column name.
        :type key: str
        """
        return self._dataflow[key]

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_by_name(workspace, name, version='latest'):
        """Get a registered Dataset from workspace by its registration name.

        :param workspace: The existing AzureML workspace in which the Dataset was registered.
        :type workspace: azureml.core.Workspace
        :param name: The registration name.
        :type name: str
        :param version: The registration version. Defaults to 'latest'.
        :type version: int
        :return: The registered dataset object.
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        dataset = AbstractDataset._get_by_name(workspace, name, version)
        AbstractDataset._track_lineage([dataset])
        return dataset

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_by_id(workspace, id):
        """Get a Dataset which is saved to the workspace.

        :param workspace: The existing AzureML workspace in which the Dataset is saved.
        :type workspace: azureml.core.Workspace
        :param id: The id of dataset.
        :type id: str
        :return: The dataset object.
            If dataset is registered, its registration name and version will also be returned.
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        dataset = AbstractDataset._get_by_id(workspace, id)
        AbstractDataset._track_lineage([dataset])
        return dataset

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_all(workspace):
        """Get all the registered datasets in the workspace.

        :param workspace: The existing AzureML workspace in which the Datasets were registered.
        :type workspace: azureml.core.Workspace
        :return: A dictionary of TabularDataset and FileDataset objects keyed by their registration name.
        :rtype: dict[str, typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]]
        """
        def list_dataset(continuation_token):
            return _restclient(workspace).dataset.list(
                subscription_id=workspace.subscription_id,
                resource_group_name=workspace.resource_group,
                workspace_name=workspace.name,
                page_size=100,
                include_latest_definition=True,
                include_invisible=False,
                continuation_token=continuation_token,
                custom_headers=_custom_headers)

        def get_dataset(name):
            try:
                return AbstractDataset.get_by_name(workspace, name)
            except Exception:
                return None

        return _DatasetDict(workspace=workspace, list_fn=list_dataset, get_fn=get_dataset)

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def id(self):
        """Return the identifier of the dataset.

        :return: Dataset id. If the dataset is not saved to any workspace, the id will be None.
        :rtype: str
        """
        if self._registration:
            if self._registration.saved_id:
                return self._registration.saved_id
            if self._registration.workspace:
                self._ensure_saved(self._registration.workspace)
                return self._registration.saved_id
        return None

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def name(self):
        """Return the registration name.

        :return: Dataset name.
        :rtype: str
        """
        return None if self._registration is None else self._registration.name

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def version(self):
        """Return the registration version.

        :return: Dataset version.
        :rtype: int
        """
        return None if self._registration is None else self._registration.version

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def description(self):
        """Return the registration description.

        :return: Dataset description.
        :rtype: str
        """
        return None if self._registration is None else self._registration.description

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def tags(self):
        """Return the registration tags.

        :return: Dataset tags.
        :rtype: str
        """
        return None if self._registration is None else self._registration.tags

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def data_changed_time(self):
        """Return the source data changed time.

        .. remarks::

            Data changed time is available for file-based data source. None will be returned when the data source is
            not supported for checking when change has happened.

        :return: The time when the most recent change happened to source data.
        :rtype: datetime.datetime
        """
        _, changed_time = self._dataflow._get_source_data_hash()
        return changed_time

    @property
    @track(_get_logger)
    def _dataflow(self):
        if self._definition is None:
            raise UserErrorException('Dataset definition is missing. Please check how the dataset is created.')
        if self._registration and self._registration.workspace:
            dataprep().api._datastore_helper._set_auth_type(self._registration.workspace)
        if not isinstance(self._definition, dataprep().Dataflow):
            try:
                self._definition = dataprep().Dataflow.from_json(self._definition)
            except Exception as e:
                msg = 'Failed to load dataset definition with azureml-dataprep=={}'.format(dataprep().__version__)
                _get_logger().error('{}. Exception: {}'.format(msg, e))
                raise UserErrorException('{}. Please install the latest version with "pip install -U '
                                         'azureml-dataprep".'.format(msg))

        # for backwards compatibility with older azureml-dataprep versions
        if not hasattr(self._definition, '_rs_dataflow_yaml'):
            self._definition._rs_dataflow_yaml = None

        return self._definition

    @track(_get_logger, activity_type=_PUBLIC_API)
    def as_named_input(self, name):
        """Provide a name for this dataset which will be used to retrieve the materialized dataset in the run.

        .. remarks::

            The name here will only be applicable inside an Azure Machine Learning run. The name must only contain
            alphanumeric and underscore characters so it can be made available as an environment variable. You can use
            this name to retrieve the dataset in the context of a run using two approaches:

            * Environment Variable:
                The name will be the environment variable name and the materialized dataset will
                be made available as the value of the environment variable. If the dataset is downloaded or mounted,
                the value will be the downloaded/mounted path. For example:

            .. code-block:: python

                # in your job submission notebook/script:
                dataset.as_named_input('foo').as_download('/tmp/dataset')

                # in the script that will be executed in the run
                import os
                path = os.environ['foo'] # path will be /tmp/dataset

            .. note::
                If the dataset is set to direct mode, then the value will be the dataset ID. You can then
                retrieve the dataset object by doing `Dataset.get_by_id(os.environ['foo'])`

            * Run.input_datasets:
                This is a dictionary where the key will be the dataset name you specified in this
                method and the value will be the materialized dataset. For downloaded and mounted dataset, the value
                will be the downloaded/mounted path. For direct mode, the value will be the same dataset object you
                specified in your job submission script.

            .. code-block:: python

                # in your job submission notebook/script:
                dataset.as_named_input('foo') # direct mode

                # in the script that will be executed in the run
                run = Run.get_context()
                run.input_datasets['foo'] # this returns the dataset object from above.


        :param name: The name of the dataset for the run.
        :type name: str
        :return: The configuration object describing how the Dataset should be materialized in the run.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        return DatasetConsumptionConfig(name, self)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def register(self, workspace, name, description=None, tags=None, create_new_version=False):
        """Register the dataset to the provided workspace.

        :param workspace: The workspace to register the dataset.
        :type workspace: azureml.core.Workspace
        :param name: The name to register the dataset with.
        :type name: str
        :param description: A text description of the dataset. Defaults to None.
        :type description: str
        :param tags: Dictionary of key value tags to give the dataset. Defaults to None.
        :type tags: dict[str, str]
        :param create_new_version: Boolean to register the dataset as a new version under the specified name.
        :type create_new_version: bool
        :return: The registered dataset object.
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        dataset = self._register(workspace, name, description, tags, create_new_version)
        self.__class__._track_output_reference_lineage(dataset)
        return dataset

    @track(_get_logger, activity_type=_PUBLIC_API)
    def update(self, description=None, tags=None):
        """Perform an in-place update of the dataset.

        :param description: The new description to use for the dataset. This description replaces the existing
            description. Defaults to existing description. To clear description, enter empty string.
        :type description: str
        :param tags: A dictionary of tags to update the dataset with. These tags replace existing tags for the
            dataset. Defaults to existing tags. To clear tags, enter empty dictionary.
        :type tags: dict[str, str]
        :return: The updated dataset object.
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        if not self._registration or not self._registration.workspace or not self._registration.registered_id:
            return UserErrorException('To update this dataset it must be registered.')
        workspace = self._registration.workspace

        def update_request():
            updated_description = description
            updated_tags = tags
            if description is None:
                updated_description = self._registration.description
            if tags is None:
                updated_tags = self._registration.tags

            return _restclient(workspace).dataset.update_dataset(
                workspace.subscription_id,
                workspace.resource_group,
                workspace.name,
                dataset_id=self._registration.registered_id,
                new_dataset_dto=_dataset_to_dto(
                    self,
                    self.name,
                    updated_description,
                    updated_tags,
                    self._registration.registered_id),
                custom_headers=self._get_telemetry_headers())

        success, result = _make_request(update_request)
        if not success:
            raise result
        result_dataset = _dto_to_dataset(workspace, result)
        self._registration.tags = result_dataset.tags
        self._registration.description = result_dataset.description
        return result_dataset

    @track(_get_logger, activity_type=_PUBLIC_API)
    def add_tags(self, tags=None):
        """Add key value pairs to the tags dictionary of this dataset.

        :param tags: The dictionary of tags to add.
        :type tags: dict[str, str]
        :return: The updated dataset object.
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        if not self._registration or not self._registration.workspace or not self._registration.registered_id:
            return UserErrorException('To add tags to this dataset it must be registered.')
        workspace = self._registration.workspace

        def add_tags_request():
            duplicate_keys = []
            for item in set(tags).intersection(self._registration.tags):
                if self._registration.tags[item] != tags[item]:
                    duplicate_keys.append(item)
            if len(duplicate_keys) > 0:
                raise UserErrorException((
                    'Dataset already contains different values for tags '
                    'with the following keys {}'
                ).format(duplicate_keys))

            updatedTags = deepcopy(self._registration.tags)
            updatedTags.update(tags)

            return _restclient(workspace).dataset.update_dataset(
                workspace.subscription_id,
                workspace.resource_group,
                workspace.name,
                dataset_id=self._registration.registered_id,
                new_dataset_dto=_dataset_to_dto(
                    self,
                    self.name,
                    self.description,
                    updatedTags,
                    self._registration.registered_id),
                custom_headers=self._get_telemetry_headers())

        success, result = _make_request(add_tags_request)
        if not success:
            raise result
        result_dataset = _dto_to_dataset(workspace, result)
        self._registration.tags = result_dataset.tags
        return result_dataset

    @track(_get_logger, activity_type=_PUBLIC_API)
    def remove_tags(self, tags=None):
        """Remove the specified keys from tags dictionary of this dataset.

        :param tags: The list of keys to remove.
        :type tags: builtin.list[str]
        :return: The updated dataset object.
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        if not self._registration or not self._registration.workspace or not self._registration.registered_id:
            return UserErrorException('To remove tags from this dataset it must be registered.')
        workspace = self._registration.workspace

        def remove_tags_request():
            updatedTags = deepcopy(self._registration.tags)
            for item in set(tags).intersection(updatedTags):
                del updatedTags[item]

            return _restclient(workspace).dataset.update_dataset(
                workspace.subscription_id,
                workspace.resource_group,
                workspace.name,
                dataset_id=self._registration.registered_id,
                new_dataset_dto=_dataset_to_dto(
                    self,
                    self.name,
                    self.description,
                    updatedTags,
                    self._registration.registered_id),
                custom_headers=self._get_telemetry_headers())

        success, result = _make_request(remove_tags_request)
        if not success:
            raise result
        dataset = _dto_to_dataset(workspace, result)
        self._registration.tags = dataset.tags
        return dataset

    @track(_get_logger, activity_type=_PUBLIC_API)
    def unregister_all_versions(self):
        """Unregister all versions under the registration name of this dataset from the workspace.

        .. remarks::

            The operation does not change any source data.
        """
        if not self._registration or not self._registration.workspace or not self._registration.registered_id:
            return  # no-op if dataset is not registered
        workspace = self._registration.workspace

        def unregister_all_versions_request():
            return _restclient(workspace).dataset.unregister_dataset(
                workspace.subscription_id,
                workspace.resource_group,
                workspace.name,
                self.name,
                custom_headers=_custom_headers)

        success, result = _make_request(unregister_all_versions_request)
        if not success:
            raise result
        self._registration = None

    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_partition_key_values(self, partition_keys=None):
        """Return unique key values of partition_keys.

        validate if partition_keys is a valid subset of full set of partition keys, return unique key values of
        partition_keys, default to return the unique key combinations by taking the full set of partition keys of this
        dataset if partition_keys is None

        .. code-block:: python

            # get all partition key value pairs
            partitions = ds.get_partition_key_values()
            # Return [{'country': 'US', 'state': 'WA', 'partition_date': datetime('2020-1-1')}]

            partitions = ds.get_partition_key_values(['country'])
            # Return [{'country': 'US'}]

        :param partition_keys: partition keys
        :type partition_keys: builtin.list[str]
        """
        import time
        starting_time = time.process_time()

        if not partition_keys:
            partition_keys = self.partition_keys

        partition_key_values = self._get_partition_key_values(get_common_path=False, partition_keys=partition_keys)

        if self._registration and self._registration.workspace:
            collect_datasets_usage(_get_logger(), _PATITION_KEY_VALUES_ACTIVITY,
                                   [self], self._registration.workspace, "{}",
                                   {"execution_time": time.process_time() - starting_time,
                                    "number_of_partition_keys": len(partition_keys)})
        return partition_key_values

    @track(_get_logger)
    def _get_partition_key_values_with_common_path(self, partition_keys=None):
        """Return unique key values of partition_keys + common path for the files.

        Intended for internal use to optimize retrieving partitions when the partitioned dataset has a
        well structured directory where we can use the returned common path to narrow down the
        scope when listing the parent directory to obtain information about the files in it.

        Can use the common path that is returned to call _get_partition_using_common_path()

        .. code-block:: python

            partition_kv, common_path = ds._get_partition_key_values_with_common_path(['country'])
            # Return [{'country': 'US', 'azureml_common_path': 'mypartitiondata/projects/computer-vision'}]

        :param partition_keys: partition keys
        :type partition_keys: builtin.list[str]
        :return: list of dicts containing partition_key_values, common_path
        :rtype: list[dict{partition_key_values, common_path}]
        """
        import time
        starting_time = time.process_time()

        if not partition_keys:
            partition_keys = self.partition_keys

        from azureml.dataprep import ExecutionError
        success = False
        try:
            partition_key_values_with_common_path = self._get_partition_key_values(get_common_path=True,
                                                                                   partition_keys=partition_keys)
            success = True
        except ExecutionError as e:
            _get_logger().warn("Encountered an error while retrieving common_path for optimization." + str(e))
            partition_key_values_with_common_path = self._get_partition_key_values(get_common_path=False,
                                                                                   partition_keys=partition_keys)

        result = []
        for kv_path_dict in partition_key_values_with_common_path:
            if success:
                common_path = kv_path_dict[_COMMON_PATH_SUMMARY_COLUMN_NAME]
                del kv_path_dict[_COMMON_PATH_SUMMARY_COLUMN_NAME]
            else:
                # failed to get common path, falling back to just retrieving partition key values
                common_path = ""
            partition_kv_with_path = _PartitionKeyValueCommonPath(kv_path_dict, common_path)
            result.append(partition_kv_with_path)

        if self._registration and self._registration.workspace:
            collect_datasets_usage(_get_logger(), _PATITION_KEY_VALUES_ACTIVITY,
                                   [self], self._registration.workspace, "{}",
                                   {"execution_time": time.process_time() - starting_time,
                                    "number_of_partition_keys": len(partition_keys)})
        return result

    @track(_get_logger)
    def _get_partition_using_partition_key_values_common_path(self, partition_key_values_common_path, verbose=False):
        """Return new partition dataset using the common path for optimization.

        Intended for internal use to retrieve a single partition from an existing partitioned dataset
        using the partition key values and the common path that was obtained from observing the
        underlying file directory structure for the partitioned dataset.

        :param partition_key_values_common_path: list of _PartitionKeyValueCommonPath objects
            ex)
                _PartitionKeyValueCommonPath.key_values = [{'file_name': 'cat1', 'file_type': 'cats'}]
                _PartitionKeyValueCommonPath.common_path = 'mydatastore/cats/'
        :type partition_key_values_common_path: builtin.list[_PartitionKeyValueCommonPath]
        :return: A new dataset object containing the partition
        :rtype: typing.Union[azureml.data.TabularDataset, azureml.data.FileDataset]
        """
        if not self.partition_keys or len(self.partition_keys) == 0:
            raise UserErrorException("_get_partition_using_partition_key_values_common_path is not available "
                                     "to a dataset that has no partition keys")

        partition_kv_dict = partition_key_values_common_path.key_values

        if partition_key_values_common_path.common_path:
            common_path = partition_key_values_common_path.common_path

            new_datasource_step = deepcopy(self._dataflow._steps[0])
            step_type = new_datasource_step.step_type
            step_arguments = new_datasource_step.arguments

            if hasattr(step_arguments, 'to_pod'):
                step_arguments = step_arguments.to_pod()

            # remove leading forward slash to normalize path segments
            common_path = common_path.lstrip('/')
            common_path_list = common_path.split('/')

            if step_type == 'Microsoft.DPrep.GetDatastoreFilesBlock':
                stored_datastores = step_arguments['datastores']
                for store in stored_datastores:
                    stored_datastore = store['datastoreName']
                    stored_path_on_datastore = store['path'].lstrip('/')
                    datastore = common_path_list[0]
                    if stored_datastore != datastore:
                        # we know it doesn't match what we need, so remove the stream_info and continue
                        del store
                        continue
                    store['path'] = _get_optimal_path(common_path_list[1:],
                                                      stored_path_on_datastore.split('/'), verbose)
            elif step_type == 'Microsoft.DPrep.GetFilesBlock':
                details = step_arguments['path']['resourceDetails']
                for detail in details:
                    stored_path = detail['path'].lstrip('/')
                    detail['path'] = _get_optimal_path(common_path_list, stored_path.split('/'), verbose)
            elif step_type == 'Microsoft.DPrep.CreateDatasetBlock':
                stream_infos = step_arguments['streamInfos']
                for stream_info in stream_infos:
                    stored_datastore = stream_info['arguments']['datastoreName']
                    stored_path_on_datastore = stream_info['resourceidentifier'].lstrip('/')
                    datastore = common_path_list[0]
                    if stored_datastore != datastore:
                        # we know it doesn't match what we need, so remove the stream_info and continue
                        del stream_info
                        continue
                    stream_info['resourceidentifier'] = _get_optimal_path(common_path_list[1:],
                                                                          stored_path_on_datastore.split('/'), verbose)
            elif step_type == 'Microsoft.DPrep.CreateDatasetFilesBlock':
                details = step_arguments['path']['resourceDetails']
                for detail in details:
                    stored_path = detail['path'].lstrip('/')
                    detail['path'] = _get_optimal_path(common_path_list, stored_path.split('/'), verbose)
            else:
                _get_logger().warning("cannot optimize as step type: {} is not one of "
                                      "GetDatastoreFilesBlock, GetFilesBlock, CreateDatasetBlock, "
                                      "and CreateDatasetFilesBlock".format(step_type))
        else:
            _get_logger().warning("{} has been provided as the 'common_path'. "
                                  "Falling back to using filter() to retrieve "
                                  "each partition.".format("None" if partition_key_values_common_path.common_path
                                                           is None else "An empty string"))

        from azureml.dataprep import col
        filter_condition = None
        for k, v in partition_kv_dict.items():
            new_filter_condition = col(k) == v
            filter_condition = new_filter_condition if filter_condition is None \
                else filter_condition & new_filter_condition

        if partition_key_values_common_path.common_path:
            new_dataflow = dataprep().Dataflow(self._dataflow._engine_api,
                                               [new_datasource_step] + self._dataflow._steps[1:])
            new_dataflow = new_dataflow.filter(filter_condition)
        else:
            new_dataflow = self._dataflow.filter(filter_condition)
        filtered_dataset = self._create(new_dataflow, self._properties, telemetry_info=self._telemetry_info)
        return filtered_dataset

    def _get_partition_key_values(self, get_common_path=False, partition_keys=None):
        if not self.partition_keys or len(self.partition_keys) == 0:
            raise UserErrorException("cannot retrieve partition key values for a dataset that has no "
                                     "partition keys")

        invalid_keys = []
        for key in partition_keys:
            if key not in self.partition_keys:
                invalid_keys.append(key)
        if len(invalid_keys) != 0:
            raise UserErrorException("{0} are invalid partition keys".format(invalid_keys))

        from azureml.dataprep.api.functions import get_stream_properties

        if not get_common_path:
            dataflow = self._dataflow.filter(get_stream_properties(self._dataflow['Path'])['Size'] > 0)
            dataflow = dataflow.keep_columns(partition_keys)
            _remove_data_read_steps_and_drop_columns(dataflow, True)
            dataflow = dataflow.distinct_rows()
            pd = dataflow.to_pandas_dataframe()
            partition_key_values = pd.to_dict(orient='records') if pd.shape[0] != 0 else []
            return partition_key_values
        else:
            import azureml.dataprep as dprep
            dataflow = self._dataflow.filter(get_stream_properties(self._dataflow['Path'])['Size'] > 0)
            dataflow = dataflow.summarize(
                summary_columns=[
                    dprep.SummaryColumnsValue(
                        column_id=_PATH_COLUMN_NAME,
                        summary_column_name=_COMMON_PATH_SUMMARY_COLUMN_NAME,
                        summary_function=dprep.SummaryFunction.COMMONPATH)
                ],
                group_by_columns=partition_keys
            )
            _remove_data_read_steps_and_drop_columns(dataflow, True)

            pd = dataflow.to_pandas_dataframe()
            partition_key_values_with_common_path = pd.to_dict(orient='records') if pd.shape[0] != 0 else []
            return partition_key_values_with_common_path

    @staticmethod
    @track(_get_logger, activity_type=_INTERNAL_API)
    def _load(path: str, workspace):
        AbstractDataset._validate_args(path, workspace)

        from azureml.dataprep.api.mltable._mltable_helper import _download_mltable_yaml, _is_tabular,\
            _parse_path_format, _PathType, UserErrorException as MLtable_UserErrorException
        from azureml.dataprep.api._rslex_executor import get_rslex_executor
        from azureml.dataprep.api.errorhandlers import ValidationError
        get_rslex_executor()

        path_type, path, optional_version = _parse_path_format(path)
        local_path = path
        if path_type is _PathType.legacy_dataset:
            if isinstance(optional_version, tuple):
                # optional_version is (dataset_name, version)
                return AbstractDataset._get_by_name(workspace, optional_version[0], optional_version[1])
            else:
                # path is legacy dataset name, optional_version is the version number
                return AbstractDataset._get_by_name(workspace, path, optional_version)
        elif path_type is _PathType.cloud:
            try:
                local_path = _download_mltable_yaml(path)
            except MLtable_UserErrorException as exc:
                _get_logger().warning('Failed to download mltable yaml with error {}'.format(exc))
                raise UserErrorException("Failed to download mltable yaml with error: {}".format(exc))

        import os.path
        local_yaml_path = "{}/MLTable".format(local_path.rstrip("/"))
        if not os.path.exists(local_yaml_path):
            _get_logger().error("Not able to find MLTable file")
            raise UserErrorException('Not able to find MLTable file')

        import yaml
        from azureml.dataprep.api.dataflow import Dataflow
        with open(local_yaml_path, "r") as stream:
            try:
                mltable_yml = yaml.safe_load(stream)
                if mltable_yml is None or len(mltable_yml) == 0:
                    _get_logger().error("mltable_yaml can't be empty")
                    raise UserErrorException("mltable_yaml can't be empty")

                if type(mltable_yml) is not dict:
                    _get_logger().error("mltable_yaml is invalid. It needs to be type of dictiory")
                    raise UserErrorException("mltable_yaml is invalid. It needs to be type of dictiory")

                # keep only paths and transformations
                filtered_mltable_yml = AbstractDataset._remove_properties_not_required(mltable_yml)
                from azureml.dataprep.api.engineapi.api import get_engine_api

                # workspace information is required for now
                # but need code change in converter to make it work with short form datastore uri
                msg_args = {
                    "SubscriptionId": workspace.subscription_id,
                    "ResourceGroup": workspace.resource_group,
                    "Workspace": workspace.name,
                    "RsflowYaml": yaml.dump(filtered_mltable_yml),
                    "BasePath": path
                }
                activity_data = get_engine_api().convert_rsflow_to_dataflow_json(msg_args)
                dataflow = Dataflow._dataflow_from_activity_data(activity_data, get_engine_api())

                if AbstractDataset._should_auto_inference(filtered_mltable_yml, activity_data):
                    dataflow = AbstractDataset._learn_from_auto_type_inference(dataflow)

                from azureml.data.tabular_dataset import TabularDataset
                from azureml.data.file_dataset import FileDataset
                if _is_tabular(filtered_mltable_yml):
                    return TabularDataset._create(definition=dataflow)
                else:
                    return FileDataset._create(definition=dataflow)
            except yaml.YAMLError as exc:
                _get_logger().warning('Failed to load mltable to dataflow with yaml error {}'.format(exc))
                raise UserErrorException("MLTable yaml is invalid: {}".format(mltable_yml))
            except UserErrorException as exc:
                _get_logger().warning('Failed to load mltable to dataset with exception {}'.format(exc))
                raise UserErrorException("MLTable yaml is invalid: {}".format(mltable_yml))
            except ValidationError as exc:
                _get_logger().warning(
                    'Failed to load mltable to dataset because of invalid mltable {}'.format(exc))
                raise UserErrorException("MLTable yaml schema is invalid: {}".format(exc))
            except Exception as ex:
                if hasattr(ex, "error_code"):
                    errorCode = getattr(ex, 'error_code')
                    if errorCode == "ScriptExecution.StreamAccess.Validation":
                        _get_logger().warning(f'Failed to load mltable. '
                                              f'Validation Error Code:{ex.validation_error_code};'
                                              f'Validation Target: {ex.validation_target}')
                        raise UserErrorException("Failed to load mltable because of: {}".format(ex))
                    elif errorCode == "ScriptExecution.StreamAccess.NotFound":
                        _get_logger().warning(f'Failed to load mltable to dataset'
                                              f'because of Stream NotFound: {ex}')
                        raise UserErrorException("MLTable Stream is NotFound: {}".format(ex))
                else:
                    _get_logger().error('Failed to load mltable to dataset with exception {}'.format(ex))
                    raise Exception("Failed to load mltable to dataset with exception: {}".format(ex))

    @staticmethod
    def _remove_properties_not_required(mltable_yaml_dict):
        filtered_mltable_yaml_dict = {k: v for k, v in mltable_yaml_dict.items() if k in ['paths', 'transformations']}
        return filtered_mltable_yaml_dict

    @staticmethod
    def _learn_from_auto_type_inference(dataflow):
        column_types_builder = dataflow.builders.set_column_types()
        column_types_builder.learn()
        if len(column_types_builder.ambiguous_date_columns) > 0:
            _get_logger()\
                .warning('Ambiguous datetime formats inferred for columns {} are resolved as "month-day"'
                         .format(column_types_builder.ambiguous_date_columns))
            column_types_builder.ambiguous_date_conversions_keep_month_day()

        return column_types_builder.to_dataflow()

    @staticmethod
    def _should_auto_inference(mltable_yaml, activity_data):
        infer_column_types_key = "infer_column_types"
        if activity_data.meta is not None and infer_column_types_key in activity_data.meta:
            return activity_data.meta.get(infer_column_types_key) == "True"
        # default to true if converter doesnt return infer_column_types in meta
        elif 'transformations' in mltable_yaml.keys():
            for elem in mltable_yaml['transformations']:
                if isinstance(elem, dict) and 'read_delimited' in elem:
                    return True
                if isinstance(elem, str) and elem == 'read_delimited':
                    return True

        return False

    @staticmethod
    def _validate_args(path: str, workspace):
        if path is None or len(path) == 0:
            raise UserErrorException("path can't be empty")

        if workspace is None:
            raise UserErrorException("workspace is required")

    @track(_get_logger)
    def _validate_path_is_stream_info(self, stream_column):
        FieldType = dataprep().api.engineapi.typedefinitions.FieldType
        try:
            stream_column_type = self._dataflow.keep_columns([stream_column]).take(1).dtypes[stream_column]
        except KeyError:
            raise UserErrorException('Column: {} is not present in the dataset. Please '
                                     'ensure that the column exists.'.format(stream_column))
        if stream_column_type != FieldType.STREAM:
            raise UserErrorException('Please make sure the stream_column: {} is exclusively of type '
                                     'streamInfo. Current type is {}'.format(stream_column, stream_column_type))

    @classmethod
    @track(_get_logger)
    def _create(cls, definition, properties=None, registration=None, telemetry_info=None):
        if registration is not None and not isinstance(registration, _DatasetRegistration):
            raise UserErrorException('registration must be instance of `_DatasetRegistration`')
        if telemetry_info is not None and not isinstance(telemetry_info, _DatasetTelemetryInfo):
            raise UserErrorException('telemetry_info must be instance of `_DatasetTelemetryInfo`')
        dataset = cls()
        dataset._definition = definition  # definition is either str or Dataflow which is immutable
        dataset._properties = deepcopy(properties) if properties else {}
        dataset._registration = registration
        dataset._telemetry_info = telemetry_info
        return dataset

    @track(_get_logger)
    def _ensure_saved(self, workspace):
        saved_id = self._ensure_saved_internal(workspace)
        self.__class__._track_output_reference_lineage(self)
        return saved_id

    def _get_telemetry_headers(self):
        telemetry_entry_point = _TELEMETRY_ENTRY_POINT_DATASET
        if self._telemetry_info and self._telemetry_info.entry_point:
            telemetry_entry_point = self._telemetry_info.entry_point
        headers = {'x-ms-azureml-dataset-entry-point': telemetry_entry_point}
        headers.update(_custom_headers)
        return headers

    @staticmethod
    def _track_lineage(datasets):
        from azureml.core import Run

        try:
            run = Run.get_context(allow_offline=False)
            run._update_dataset_lineage(datasets)
        except RunEnvironmentException:
            # offline run nothing to do
            pass
        except AttributeError:
            pass
        except Exception:
            _get_logger().error('Failed to update dataset lineage')

    @staticmethod
    def _track_output_reference_lineage(dataset):
        from azureml._restclient.models import OutputDatasetLineage, DatasetIdentifier, DatasetOutputType
        from azureml.core import Run

        if not dataset._registration:
            return

        id = dataset.id
        registered_id = dataset._registration and dataset._registration.registered_id
        version = dataset.version
        dataset_id = DatasetIdentifier(id, registered_id, version)
        output_lineage = OutputDatasetLineage(dataset_id, DatasetOutputType.reference)

        try:
            run = Run.get_context(allow_offline=False)
            run._update_output_dataset_lineage([output_lineage])
        except RunEnvironmentException:
            # offline run nothing to do
            pass
        except AttributeError:
            pass
        except Exception as e:
            _get_logger().warning('Failed to update output dataset lineage due to {}.'.format(type(e).__name__))

    @staticmethod
    def _get_by_name(workspace, name, version):
        if version != 'latest' and version is not None:
            try:
                version = int(version)
            except Exception:
                raise UserErrorException('Invalid value {} for version. Version value must be number or "latest".'
                                         .format(version))
        else:
            version = None

        def get_by_name_request():
            return _restclient(workspace).dataset.get_dataset_by_name(workspace.subscription_id,
                                                                      workspace.resource_group,
                                                                      workspace.name,
                                                                      dataset_name=name,
                                                                      version_id=version,
                                                                      custom_headers=_custom_headers)

        def handle_error(error):
            if error.response.status_code == 404:
                return UserErrorException(
                    'Cannot find dataset registered with name "{}"{} in the workspace.'
                    .format(name, '' if version == 'latest' else ' (version: {})'.format(version)))

        success, result = _make_request(get_by_name_request, handle_error)
        if not success:
            raise result

        dataset = _dto_to_dataset(workspace, result)
        warn_deprecated_blocks(dataset)
        return dataset

    @staticmethod
    def _get_by_id(workspace, id):
        def get_by_id_request_for_registered():
            return _restclient(workspace).dataset.get_datasets_by_saved_dataset_id(
                subscription_id=workspace.subscription_id,
                resource_group_name=workspace.resource_group,
                workspace_name=workspace.name,
                saved_dataset_id=id,
                # just need the 1st (can only be more than one for dataset created in the old age)
                page_size=1,
                custom_headers=_custom_headers)

        success, result = _make_request(get_by_id_request_for_registered)

        if success and len(result.value) == 1:
            dataset = _dto_to_dataset(workspace, result.value[0])
        else:
            def get_by_id_request_for_unregistered():
                return _restclient(workspace).dataset.get_by_id(
                    subscription_id=workspace.subscription_id,
                    resource_group_name=workspace.resource_group,
                    workspace_name=workspace.name,
                    id=id,
                    resolve_legacy_id=True,
                    custom_headers=_custom_headers)

            def handle_error(error):
                if error.response.status_code == 404:
                    return UserErrorException(
                        'Cannot find dataset with id "{}" in the workspace.'
                        .format(id))

            success, result = _make_request(get_by_id_request_for_unregistered, handle_error, 20)
            if not success:
                raise result

            dataset = _saved_dataset_dto_to_dataset(workspace, result)

        warn_deprecated_blocks(dataset)
        return dataset

    def _register(self, workspace, name, description=None, tags=None, create_new_version=False):
        def register_request():
            return _restclient(workspace).dataset.register(
                workspace.subscription_id,
                workspace.resource_group,
                workspace.name,
                dataset_dto=_dataset_to_dto(self, name, description, tags, create_new_version=create_new_version),
                if_exists_ok=create_new_version,
                update_definition_if_exists=create_new_version,
                custom_headers=self._get_telemetry_headers())

        def handle_error(error):
            status_code = error.response.status_code
            if status_code == 409:
                return UserErrorException((
                    'There is already a dataset registered under name "{}". '
                    'Specify `create_new_version=True` to register the dataset as a new version. '
                    'Use `update`, `add_tags`, or `remove_tags` to change only the description or tags.'
                ).format(name))
            if status_code == 400:
                regex = re.compile(
                    r'has been registered as (.+):([0-9]+) \(name:version\).',
                    re.IGNORECASE)
                matches = regex.findall(error.message)
                if len(matches) == 1:
                    existing_name, existing_version = matches[0]
                    return UserErrorException((
                        'An identical dataset had already been registered, which can '
                        + 'be retrieved with `Dataset.get_by_name(workspace, name="{}", version={})`.'
                    ).format(existing_name, existing_version))

        success, result = _make_request(register_request, handle_error)
        if not success:
            raise result

        return _dto_to_dataset(workspace, result)

    def _ensure_saved_internal(self, workspace):
        if not self._registration or not self._registration.saved_id:
            # only call service when dataset is not saved yet
            def ensure_saved_request():
                return _restclient(workspace).dataset.ensure_saved(
                    subscription_id=workspace.subscription_id,
                    resource_group_name=workspace.resource_group,
                    workspace_name=workspace.name,
                    dataset=_dataset_to_saved_dataset_dto(self),
                    custom_headers=self._get_telemetry_headers())

            success, result = _make_request(ensure_saved_request)

            if not success:
                raise result
            saved_dataset = _saved_dataset_dto_to_dataset(workspace, result)

            # modify _definition using service response
            self._definition = saved_dataset._definition

            # modify self._registration.saved_id using service response
            if self._registration:
                self._registration.saved_id = saved_dataset._registration.saved_id
            else:
                self._registration = saved_dataset._registration

        return self._registration.saved_id

    @track(_get_logger, activity_type=_PUBLIC_API)
    def __str__(self):
        """Format the dataset object into a string.

        :return: Return string representation of the dataset object
        :rtype: str
        """
        return '{}\n{}'.format(type(self).__name__, self.__repr__())

    @track(_get_logger, activity_type=_PUBLIC_API)
    def __repr__(self):
        """Format the dataset object into a string.

        :return: Return string representation of the the dataset object
        :rtype: str
        """
        content = collections.OrderedDict()
        if is_dataprep_installed():
            steps = self._dataflow._get_steps()
            step_type = steps[0].step_type
            step_arguments = steps[0].arguments

            if hasattr(step_arguments, 'to_pod'):
                step_arguments = step_arguments.to_pod()
            if step_type == 'Microsoft.DPrep.GetDatastoreFilesBlock':
                source = ['(\'{}\', \'{}\')'.format(store['datastoreName'], store['path'])
                          for store in step_arguments['datastores']]
            elif step_type == 'Microsoft.DPrep.GetFilesBlock':
                source = [details['path'] for details in step_arguments['path']['resourceDetails']]
            elif step_type == 'Microsoft.DPrep.CreateDatasetBlock':
                source = ['(\'{}\', \'{}\')'.format(si['arguments']['datastoreName'], si['resourceidentifier'])
                          for si in step_arguments['streamInfos']]
            elif step_type == 'Microsoft.DPrep.CreateDatasetFilesBlock':
                source = [details['path'] for details in step_arguments['path']['resourceDetails']]
            else:
                source = None

            encoder = dataprep().api.engineapi.typedefinitions.CustomEncoder \
                if hasattr(dataprep().api.engineapi.typedefinitions, 'CustomEncoder') \
                else dataprep().api.engineapi.engine.CustomEncoder
            content['source'] = source
            content['definition'] = [_get_step_name(s.step_type) for s in steps]
        else:
            encoder = None
            global _dataprep_missing_for_repr_warned
            if not _dataprep_missing_for_repr_warned:
                _dataprep_missing_for_repr_warned = True
                import logging
                logging.getLogger().warning(get_dataprep_missing_message(
                    'Warning: Cannot load "definition" and "source" for the dataset'))

        if self._registration is not None:
            content['registration'] = collections.OrderedDict([
                ('id', self.id),
                ('name', self.name),
                ('version', self.version)
            ])

            if self.description:
                content['registration']['description'] = self.description
            if self.tags:
                content['registration']['tags'] = self.tags
            content['registration']['workspace'] = self._registration.workspace.__repr__()

        return json.dumps(content, indent=2, cls=encoder)

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def partition_keys(self):
        """Return the partition keys.

        :return: the partition keys
        :rtype: builtin.list[str]
        """
        from azureml.data._partition_format import parse_partition_format

        steps = self._dataflow._get_steps()
        for step in steps:
            if step.step_type == 'Microsoft.DPrep.AddColumnsFromPartitionFormatBlock' and \
                    step.arguments['partitionFormat']:
                parsed_result = parse_partition_format(step.arguments['partitionFormat'])
                if len(parsed_result) == 3 and parsed_result[2]:
                    if self._registration and self._registration.workspace:
                        collect_datasets_usage(_get_logger(), _PATITION_KEY_ACTIVITY,
                                               [self], self._registration.workspace, "N/A",
                                               {"number_of_partition_keys": len(parsed_result[2])})
                    return parsed_result[2]
        return []


class _DatasetRegistration(object):
    def __init__(self, workspace, saved_id, registered_id=None, name=None, version=None, description=None, tags=None):
        # Deep copy has been overridden, if you add any new members, please make sure you update the deepcopy method
        # accordingly
        self.workspace = workspace  # this member will not be deep copied
        self.saved_id = saved_id
        self.registered_id = registered_id
        self.name = name
        self.version = version
        self.description = description
        self.tags = tags

    def __repr__(self):
        return "DatasetRegistration(id='{}', name='{}', version={}, description='{}', tags={})".format(
            self.saved_id, self.name, self.version, self.description or '', self.tags)

    def __deepcopy__(self, memodict={}):
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'workspace':
                # we don't deepcopy workspace because some Authentication classes has locks which cannot be deep
                # copied
                continue
            setattr(result, k, deepcopy(v))
        return result


class _DatasetTelemetryInfo(object):
    def __init__(self, entry_point):
        self.entry_point = entry_point


_Mapping = collections.abc.Mapping if hasattr(collections, 'abc') else collections.Mapping


class _DatasetDict(_Mapping):
    def __init__(self, workspace, list_fn, get_fn):
        self._workspace = workspace
        self._list_fn = list_fn
        self._get_fn = get_fn
        self._all_listed = False
        self._list_continuation_token = None
        self._list_cached = {}
        self._getitem_cached = {}

    @property
    def registrations(self):
        self._list_all()
        return list(self._list_cached.values())

    def __getitem__(self, key):
        if key in self._getitem_cached:
            return self._getitem_cached[key]
        result = self._get_fn(name=key)
        if result is None:
            raise KeyError(key)
        self._getitem_cached[key] = result
        return result

    def __iter__(self):
        return iter(self._list_cached if self._all_listed else _DatasetDictKeyIterator(self))

    def __len__(self):
        self._list_all()
        return len(self._list_cached)

    def __str__(self):
        self._list_all()
        return pprint.pformat(self._list_cached, indent=2)

    def __repr__(self):
        self._list_all()
        return str({name: self._list_cached[name] for name in self._list_cached})

    def _list_all(self):
        while not self._all_listed:
            self._list_more()

    def _list_more(self):
        new_listed_names = []
        if not self._all_listed:
            list_result = self._list_fn(continuation_token=self._list_continuation_token)
            if list_result.continuation_token is None:
                self._all_listed = True
            else:
                self._list_continuation_token = list_result.continuation_token
            for ds in list_result.value:
                if ds is not None:
                    self._list_cached[ds.name] = _dto_to_registration(self._workspace, ds)
                    new_listed_names.append(ds.name)
        return new_listed_names


class _DatasetDictKeyIterator():
    def __init__(self, ds_dict):
        self._ds_dict = ds_dict
        self._pending_keys = list(ds_dict._list_cached.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self._ds_dict._all_listed and not self._pending_keys:
            raise StopIteration
        if not self._pending_keys:
            self._pending_keys.extend(self._ds_dict._list_more())
        if not self._pending_keys:
            raise StopIteration
        return self._pending_keys.pop(0)


class _PartitionKeyValueCommonPath():
    def __init__(self, partition_kv_dict, common_path):
        self.key_values = partition_kv_dict
        self.common_path = common_path

    def __eq__(self, other):
        return (self.key_values, self.common_path) == (other.key_values, other.common_path)


def _get_path_from_step(step_type, step_arguments):
    if step_type == 'Microsoft.DPrep.GetDatastoreFilesBlock':
        datastores = step_arguments['datastores']
        if len(datastores) > 1:
            return None
        datastore = datastores[0]
        return '{}/{}'.format(datastore['datastoreName'], datastore['path'].lstrip('/\\'))
    if step_type == 'Microsoft.DPrep.GetFilesBlock':
        resource_details = step_arguments['path']['resourceDetails']
        if len(resource_details) > 1:
            return None
        return resource_details[0]['path']
    if step_type == 'Microsoft.DPrep.ReferenceAndInverseSplitBlock':
        source_step = step_arguments['sourceFilter']['anonymousSteps'][0]
        return _get_path_from_step(source_step['type'], source_step['arguments'])
    _get_logger().warning('Unrecognized type "{}" for first step in FileDataset.'.format(step_type))
    return None


def _get_step_name(step_type):
    parsed = re.match(r'Microsoft.DPrep.(?P<stepName>.*)Block', step_type, re.IGNORECASE)
    return parsed.group('stepName') if parsed else step_type


def _remove_data_read_steps_and_drop_columns(dataflow, remove_drop_column):
    for step in dataflow._steps:
        if step.step_type == 'Microsoft.DPrep.ReadParquetFileBlock' or \
                step.step_type == 'Microsoft.DPrep.ParseDelimitedBlock' or \
                step.step_type == 'Microsoft.DPrep.ParseJsonLinesBlock':
            dataflow._steps.remove(step)
        # need to remove dropColumns step if 'Path' column is going to be dropped
        if remove_drop_column and step.step_type == 'Microsoft.DPrep.DropColumnsBlock':
            step_arguments = step.arguments
            if hasattr(step_arguments, 'to_pod'):
                step_arguments = step_arguments.to_pod()
            columns = step_arguments['columns']
            if 'selectedColumn' in columns['details']:
                selected_columns = columns['details']['selectedColumn']
            else:
                selected_columns = columns['details']['selectedColumns']
            if "Path" in selected_columns:
                dataflow._steps.remove(step)


def _get_optimal_path(common_path_list, stored_path_list, verbose):
    """Merge common path segment list with stored path segment list and returns optimal path.

    :param common_path_list: list containing common path segments
    :type partition_keys: builtin.list[str]
    :param stored_path_list: list containing stored path segments
    :type stored_path_list: builtin.list[str]
    :param verbose: boolean value to print out the optimizations performed
    :type verbose: bool
    :return: str
    :rtype: str
    """
    stored_path_str = '/'.join(stored_path_list)

    # UX case - check if the path ends with /**/* -> try to normalize it to /
    try:
        if stored_path_list[-2] == "**" and stored_path_list[-1] == "*":
            del stored_path_list[-2]
            del stored_path_list[-1]
    except IndexError:
        pass

    optimal_path_list = []
    optimized_count = 0
    used_existing_count = 0

    for path_elem, stored_path_elem in itertools.zip_longest(common_path_list, stored_path_list):
        if path_elem and not stored_path_elem:
            # case where we were able to optimize using common path
            optimized_count += 1
            optimal_path_list.append(path_elem)
        elif not path_elem and stored_path_elem:
            used_existing_count += 1
            optimal_path_list.append(stored_path_elem)
        else:
            # case where both are not None
            # if a common path segment exists, it should be used as the value is obtained from
            # having observed the underlying dataset directory structure
            if path_elem != stored_path_elem:
                optimized_count += 1
                optimal_path_list.append(path_elem)
            else:
                used_existing_count += 1
                optimal_path_list.append(stored_path_elem)

    # print how many segments we were able to optimize by showing the two different path values.
    result = '/'.join(optimal_path_list)
    if verbose:
        print("Optimized {} path segments using provided common path.".format(optimized_count))
        print("Stored path: {}".format(stored_path_str))
        print("Optimized path: {}".format(result))

    _get_logger().info("Successfully optimized {} path segments and used {} existing "
                       "path segments.".format(optimized_count, used_existing_count))
    return result
