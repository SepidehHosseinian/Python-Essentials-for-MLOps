# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality to manage dataset definition and its operations.

>[!NOTE]
>This module is deprecated. For more information, see https://aka.ms/dataset-deprecation.
"""

import logging
from azureml.data._dataset_deprecation import deprecated


module_logger = logging.getLogger(__name__)

try:
    from azureml.dataprep import Dataflow

    class DatasetDefinition(Dataflow):
        """Defines a series of steps that specify how to read and transform data in a Dataset.

        .. note::

            This class is deprecated. For more information, see https://aka.ms/dataset-deprecation.

        A Dataset registered in an Azure Machine Learning workspace can have multiple definitions, each
        created by calling :func:`azureml.core.dataset.Dataset.update_definition`. Each definition has
        an unique identifier. The current definition is the latest one created.

        For unregistered Datasets, only one definition exists.

        Dataset definitions support all the transformations listed for the :class:`azureml.dataprep.Dataflow` class:
        see http://aka.ms/azureml/howto/transformdata. To learn more about Dataset Definitions, go
        to https://aka.ms/azureml/howto/versiondata.
        """

        @deprecated('DatasetDefinition class')
        def __init__(self, workspace=None, dataset_id=None, version_id=None, dataflow=None,
                     dataflow_json=None, notes=None, etag=None, created_time=None, modified_time=None,
                     state=None, deprecated_by_dataset_id=None, deprecated_by_definition_version=None,
                     data_path=None, dataset=None, file_type="Unknown"):
            """Initialize the Dataset definition object.

            :param workspace: The workspace the Dataset is registered in.
            :type workspace: str
            :param dataset_id: The Dataset identifier.
            :type dataset_id: str
            :param version_id: The definition version.
            :type version_id: str
            :param dataflow: The Dataflow object.
            :type dataflow: azureml.dataprep.Dataflow
            :param dataflow_json: The Dataflow json.
            :type dataflow: str
            :param notes: Optional information about the definition.
            :type notes: str
            :param etag: Etag.
            :type etag: str
            :param created_time: The creation time of the definition.
            :type created_time: datetime
            :param modified_time: The last modified time of the definition.
            :type modified_time: datetime
            :param deprecated_by_dataset_id: The ID of the Dataset that deprecates this definition.
            :type deprecated_by_dataset_id: str
            :param deprecated_by_definition_version: The version of the definition that deprecates this definition.
            :type deprecated_by_definition_version: str
            :param data_path: The data path.
            :type data_path: azureml.data.datapath.DataPath
            :param dataset: The parent Dataset object.
            :type dataset: azureml.core.dataset.Dataset
            """
            if dataflow is None and dataflow_json is not None:
                dataflow = Dataflow.from_json(dataflow_json)

            if dataflow is not None:
                super(DatasetDefinition, self).__init__(engine_api=dataflow._engine_api, steps=dataflow._steps)

            self._workspace = workspace
            self._dataset_id = dataset_id
            self._version_id = version_id
            self._notes = notes
            self._etag = etag
            self._created_time = created_time
            self._modified_time = modified_time
            self._state = state
            self._deprecated_by_dataset_id = deprecated_by_dataset_id
            self._deprecated_by_definition_version = deprecated_by_definition_version
            self._data_path = data_path
            self._dataset = dataset
            self._file_type = file_type

        def _set_file_type(self, file_type):
            """Set the file type in Definition.

            :param: file_type.
            :type: str
            """
            self._file_type = file_type

        def deprecate(self, deprecate_by_dataset_id, deprecated_by_definition_version=None):
            """Deprecate the Dataset, with a pointer to the new Dataset.

            .. remarks::

                Deprecated dataset definitions will log warnings when they are consumed.
                To completely block a dataset definition from being consumed, archive it.

                If a dataset definition is deprecated by accident, use :meth:`reactivate` to activate it.

            :param deprecate_by_dataset_id: The dataset ID which is responsible for the deprecation of current dataset.
            :type deprecate_by_dataset_id: uuid
            :param deprecated_by_definition_version: The dataset definition version which is responsible
                    for the deprecation of current dataset definition.
            :type deprecated_by_definition_version: str
            :return: None.
            :rtype: None
            """
            DatasetDefinition._client()._deprecate_definition(
                self,
                deprecate_by_dataset_id,
                deprecated_by_definition_version)

        def archive(self):
            """Archive the dataset definition.

            .. remarks::

                After archival, any attempt to retrieve the dataset will result in an error.
                If archived by accident, use :meth:`reactivate` to activate it.

            :return: None.
            :rtype: None
            """
            DatasetDefinition._client()._archive_definition(self)

        def reactivate(self):
            """Reactivate the dataset definition.

            Works on dataset definitions that have been deprecated or archived.

            :return: None.
            :rtype: None
            """
            DatasetDefinition._client()._reactivate_definition(self)

        def to_pandas_dataframe(self):
            """Create a Pandas dataframe by executing the transformation pipeline defined by this dataset definition.

            .. remarks::

                Return a Pandas DataFrame fully materialized in memory.

            :return: A Pandas DataFrame.
            :rtype: pandas.DataFrame
            """
            return super(DatasetDefinition, self).to_pandas_dataframe()

        def to_spark_dataframe(self):
            """Create a Spark DataFrame that can execute the transformation pipeline defined by this Dataflow.

            .. remarks::

                The Spark Dataframe returned is only an execution plan and does not actually contain any data,
                as Spark Dataframes are lazily evaluated.

            :return: A Spark DataFrame.
            :rtype: pyspark.sql.DataFrame
            """
            return super(DatasetDefinition, self).to_spark_dataframe()

        def create_snapshot(
                self,
                snapshot_name,
                compute_target=None,
                create_data_snapshot=False,
                target_datastore=None):
            """Create a snapshot of the registered Dataset.

            .. remarks::

                Snapshots capture point in time summary statistics of the underlying data
                    and an optional copy of the data itself. To learn more about creating snapshots,
                    go to https://aka.ms/azureml/howto/createsnapshots.


            :param snapshot_name: The snapshot name. Snapshot names should be unique within a Dataset.
            :type snapshot_name: str
            :param compute_target: The compute target to perform the snapshot profile creation.
                If omitted, the local compute is used.
            :type compute_target: azureml.core.compute.ComputeTarget or str
            :param create_data_snapshot: If True, a materialized copy of the data will be created.
            :type create_data_snapshot: bool
            :param target_datastore: The target datastore where to save snapshot.
                If omitted, the snapshot will be created in the default storage of the workspace.
            :type target_datastore: typing.Union[azureml.data.azure_storage_datastore.AbstractAzureStorageDatastore,
                str]
            :return: A DatasetSnapshot object.
            :rtype: azureml.data.dataset_snapshot.DatasetSnapshot
            """
            return DatasetDefinition._client().create_snapshot(
                self,
                snapshot_name,
                compute_target,
                create_data_snapshot,
                target_datastore)

        def _get_datapath(self):
            if self._dataset_id is None:
                raise Exception("Get datapath is only supported for registered datasets.")
            return DatasetDefinition._client().get_datapath(self._workspace, self._dataset_id, self._version_id)

        def __getitem__(self, key):
            """Keep columns, select columns, or select steps.

            If key is a list of string, returns a new dataflow referencing this definition with an additional
            keep_columns transformation of the columns specific in the argument.
            If key is a string, returns a column selection expression with the argument being the selected column.
            If key is a slice, returns the list of steps sliced according to the slice range.

            :param key: The columns or steps to operate on.
            :type key: str, list, or slice
            :raises KeyError: If key is not of type str or list.
            :return: The resultant object based on input.
            :rtype: typing.Union[azureml.dataprep.Dataflow, azureml.dataprep.RecordFieldExpression]
            """
            if isinstance(key, slice) or isinstance(key, str):
                return super(DatasetDefinition, self).__getitem__(key)
            elif isinstance(key, list):
                if not all(map(lambda k: isinstance(k, str), key)):
                    raise KeyError('Not all column selectors are string')
                if self._workspace:
                    return Dataflow.reference(self).keep_columns(key)
                else:
                    return self.keep_columns(key)
            else:
                raise TypeError("Invalid argument type.")

        def __repr__(self):
            """
            Representation of the object.

            :return: Return the string form of the properties in DatasetDefinition object
            :rtype: str
            """
            result = "VersionID: {}, State: {}, Created: {}, Modified: {}, Notes: {}".format(
                self._version_id,
                self._state,
                self._created_time,
                self._modified_time,
                self._notes)

            if(self._state in ['deprecated']):
                result += ", DeprecatedByDatasetID:{}, DeprecatedByDefinitionVersion: {}".format(
                    self._deprecated_by_dataset_id,
                    self._deprecated_by_definition_version)
            return result

        @staticmethod
        def _client():
            """Get a Dataset client.

            :return: Returns the client
            :rtype: DatasetClient
            """
            from ._dataset_client import _DatasetClient
            return _DatasetClient

except ImportError:
    class DatasetDefinition(object):
        """A class for Dataset definition."""

        @deprecated('DatasetDefinition class')
        def __init__(self, *args, **kwargs):
            """Initialize the Dataset definition object."""
            message = "Please install azureml-dataprep package to consume DatasetDefinition class."
            module_logger.error(message)
            raise ImportError(message)
