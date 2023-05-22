# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Manages the interaction with Azure Machine Learning Datasets.

This module provides functionality for consuming raw data, managing data, and performing actions on
data in Azure Machine Learning. Use the :class:`azureml.core.Dataset` class in this module
to create datasets along with the functionality in the :mod:`azureml.data` package, which contains
the supporting classes :class:`azureml.data.FileDataset` and :class:`azureml.data.TabularDataset`.

To get started with datasets, see the article [Add & register
datasets](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets).
"""
from azureml.data.dataset_type_definitions import (
    HistogramCompareMethod,
    PromoteHeadersBehavior,
    SkipLinesBehavior,
    FileEncoding)
from azureml.data._loggerfactory import track, _LoggerFactory
from azureml.data._dataset import _Dataset
from azureml.data._dataset_deprecation import deprecated
from azureml.data.constants import _PUBLIC_API
from collections import OrderedDict


_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


class Dataset(object):
    """Represents a resource for exploring, transforming, and managing data in Azure Machine Learning.

    A Dataset is a reference to data in a :class:`azureml.core.datastore.Datastore` or behind public web urls.

    For methods deprecated in this class, please check :class:`azureml.data.abstract_dataset.AbstractDataset`
    class for the improved APIs.

    The following Datasets types are supported:

    * :class:`azureml.data.TabularDataset` represents data in a tabular format created by parsing the
      provided file or list of files.

    * :class:`azureml.data.FileDataset` references single or multiple files in datastores
      or from public URLs.

    To get started with datasets, see the article `Add & register
    datasets <https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets>`_,
    or see the notebooks https://aka.ms/tabulardataset-samplenotebook and https://aka.ms/filedataset-samplenotebook.

    .. remarks::

        The Dataset class exposes two convenience class attributes (``File`` and ``Tabular``) you
        can use for creating a Dataset without working with the corresponding factory methods. For
        example, to create a dataset using these attributes:

        * ``Dataset.Tabular.from_delimited_files()``

        * ``Dataset.File.from_files()``

        You can also create a new TabularDataset or FileDataset by directly calling the corresponding
        factory methods of the class defined in :class:`azureml.data.dataset_factory.TabularDatasetFactory` and
        :class:`azureml.data.dataset_factory.FileDatasetFactory`.

        The following example shows how to create a TabularDataset pointing to a single path in a datastore.

        .. code-block:: python

            from azureml.core import Dataset
            dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, 'train-dataset/tabular/iris.csv')])

            # preview the first 3 rows of the dataset
            dataset.take(3).to_pandas_dataframe()

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/work-with-data/datasets-tutorial/train-with-datasets/train-with-datasets.ipynb


    :var azureml.core.Dataset.File: A class attribute that provides access to the
        FileDatasetFactory methods for creating new
        FileDataset objects. Usage: Dataset.File.from_files().
    :vartype File: azureml.data.dataset_factory.FileDatasetFactory
    :var azureml.core.Dataset.Tabular: A class attribute that provides access to the
        TabularDatasetFactory methods for creating new
        TabularDataset objects. Usage: Dataset.Tabular.from_delimited_files().
    :vartype Tabular: azureml.data.dataset_factory.TabularDatasetFactory
    """

    """Factory for creating :class:`azureml.data.TabularDataset`"""
    Tabular = _Dataset.Tabular

    """Factory for creating :class:`azureml.data.FileDataset`"""
    File = _Dataset.File

    get_by_name = _Dataset.get_by_name
    get_by_id = _Dataset.get_by_id
    get_all = _Dataset.get_all
    _load = _Dataset._load

    @deprecated(
        'The constructor of Dataset',
        'factory methods from "Dataset.Tabular" and "Dataset.File" to create dataset instances')
    def __init__(self, definition, workspace=None, name=None, id=None):
        """Initialize the Dataset object.

        To obtain a Dataset that has already been registered with the workspace, use the get method.

        :param definition: The Dataset definition.
        :type definition: azureml.data.DatasetDefinition
        :param workspace: The workspace in which the Dataset exists.
        :type workspace: azureml.core.Workspace
        :param name: The name of the Dataset.
        :type name: str
        :param id: The unique identifier of the Dataset.
        :type id: str
        :return: The corresponding Dataset for the definition.
        :rtype: azureml.core.dataset.Dataset
        """
        self._definition = definition
        self._workspace = workspace
        self._name = name
        self._id = id
        self._is_visible = True
        self._tags = {}
        self._description = None
        self._state = None
        self._deprecated_by_dataset_id = None
        self._deprecated_by_definition = None
        self._created_time = None
        self._modified_time = None

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def name(self):
        """Return the Dataset name.

        :return: The Dataset name.
        :rtype: str
        """
        return self._name

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def workspace(self):
        """If the Dataset was registered in a workspace, return that. Otherwise, return None.

        :return: The workspace.
        :rtype: azureml.core.Workspace
        """
        return self._workspace

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def id(self):
        """If the Dataset was registered in a workspace, return the ID of the Dataset. Otherwise, return None.

        :return: The Dataset ID.
        :rtype: str
        """
        return self._id

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def definition_version(self):
        """Return the version of the current definition of the Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            A Dataset definition is a series of steps that specify how to read and transform data.

            A Dataset registered in an AzureML workspace can have multiple definitions, each created by calling
            :meth:`azureml.core.dataset.Dataset.update_definition`. Each definition has an unique identifier. The
            current definition is the latest one created, whose ID is returned by this.

            For unregistered Datasets, only one definition exists.

        :return: The Dataset definition version.
        :rtype: str
        """
        if self.definition is None:
            return None
        return self.definition._version_id

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def definition(self):
        """Return the current Dataset definition.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            A Dataset definition is a series of steps that specify how to read and transform data.

            A Dataset registered in an AzureML workspace can have multiple definitions, each created by calling
            :meth:`azureml.core.dataset.Dataset.update_definition`. Each definition has an unique identifier. Having
            multiple definitions allows you to make changes to existing Datasets without breaking models and
            pipelines that depend on the older definition.

            For unregistered Datasets, only one definition exists.

        :return: The Dataset definition.
        :rtype: azureml.data.dataset_definition.DatasetDefinition
        """
        if self._definition is None and self.id is not None:
            self._definition = self.get_definition()
        return self._definition

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def is_visible(self):
        """Control the visibility of a registered Dataset in the Azure ML workspace UI.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Values returned:

            * True: Dataset is visible in workspace UI. Default.
            * False: Dataset is hidden in workspace UI.

            Has no effect on unregistered Datasets.

        :return: The Dataset visibility.
        :rtype: bool
        """
        return self._is_visible

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def tags(self):
        """Return the tags associated with the Dataset.

        :return: Dataset tags.
        :rtype: dict[str, str]
        """
        return self._tags

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def description(self):
        """Return the description of the Dataset.

        .. remarks::

            Specifying a description of the data in the Dataset enables users of the workspace to
            understand what the data represents, and how they can use it.

        :return: The Dataset description.
        :rtype: str
        """
        return self._description

    @property
    @track(_get_logger, activity_type=_PUBLIC_API)
    def state(self):
        """Return the state of the Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            The meaning and effect of states are as follows:

            * Active. Active definitions are exactly what they sound like, all actions can be
              performed on active definitions.
            * Deprecated. deprecated definition can be used, but will result in a warning being
              logged in the logs everytime the underlying data is accessed.
            * Archived. An archived definition cannot be used to perform any action. To perform
              actions on an archived definition, it must be reactivated.

        :return: The Dataset state.
        :rtype: str
        """
        return self._state

    @staticmethod
    @deprecated(
        '"Dataset.get"',
        '"Dataset.get_by_name" and "Dataset.get_by_id" to retrieve dataset')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def get(workspace, name=None, id=None):
        r"""Get a Dataset that already exists in the workspace by specifying either its name or ID.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use \
            :meth:`azureml.core.Dataset.get_by_name` and :meth:`azureml.core.Dataset.get_by_id` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            You can provide either ``name`` or ``id``. An exception is raised if:

            * both ``name`` and ``id`` are specified but don't match.
            * the Dataset with the specified ``name`` or ``id`` cannot be found in the workspace.

        :param workspace: The existing AzureML workspace in which the Dataset was created.
        :type workspace: azureml.core.Workspace
        :param name: The name of the Dataset to be retrieved.
        :type name: str
        :param id: A unique identifier of the Dataset in the workspace.
        :type id: str
        :return: The Dataset with the specified name or ID.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().get(workspace, name, id)

    @staticmethod
    @deprecated(
        '"Dataset.list"',
        '"Dataset.get_all" to retrieve datasets registered in the workspace')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def list(workspace):
        r"""List all the Datasets in the workspace, including ones with ``is_visible`` property equal to False.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use :meth:`azureml.core.Dataset.get_all` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        :param workspace: The workspace for which you want to retrieve the list of Datasets.
        :type workspace: azureml.core.Workspace
        :return: A list of Dataset objects.
        :rtype: builtin.list[azureml.core.dataset.Dataset]
        """
        return Dataset._client().list(workspace)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_definitions(self):
        """Get all the definitions of the Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            A Dataset registered in an AzureML workspace can have multiple definitions, each created by calling
            :meth:`azureml.core.dataset.Dataset.update_definition`. Each definition has an unique identifier. The
            current definition is the latest one created.

            For unregistered Datasets, only one definition exists.

        :return: A dictionary of Dataset definitions.
        :rtype: dict[str, azureml.data.dataset_definition.DatasetDefinition]
        """
        if self.id is None:
            return [self.definition]
        return Dataset._client().get_dataset_definitions(self)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_definition(self, version_id=None):
        """Get a specific definition of the Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            If ``version_id`` is provided, then Azure Machine Learning tries to get the definition corresponding
            to that version. If that version does not exist, an exception is thrown.
            If ``version_id`` is omitted, then the latest version is retrieved.

        :param version_id: The version ID of the Dataset definition
        :type version_id: str
        :return: The Dataset definition.
        :rtype: azureml.data.dataset_definition.DatasetDefinition
        """
        if self.id is None:
            return self.definition

        return Dataset._client().get_dataset_definition(self, version_id)

    @staticmethod
    @deprecated(
        '"Dataset.from_pandas_dataframe"',
        '"Dataset.Tabular.register_pandas_dataframe" instead')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_pandas_dataframe(
            dataframe,
            path=None,
            in_memory=False):
        r"""Create an unregistered, in-memory Dataset from a pandas dataframe.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use ``Dataset.Tabular.register_pandas_dataframe`` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::
            Use this method to convert a Pandas dataframe to a Dataset object.
            A Dataset created by this method can not be registered, as the data is from memory.

            If ``in_memory`` is False, the Pandas DataFrame is converted to a CSV file locally. If ``pat` is of type
            DataReference, then the Pandas frame will be uploaded to the data store, and the Dataset will be based
            off the DataReference. If ``path`` is a local folder, the Dataset will be created off of the local file
            which cannot be deleted.

            Raises an exception if the current DataReference is not a folder path.

        :param dataframe: The Pandas DataFrame.
        :type dataframe: pandas.DataFrame
        :param path: A data path in registered datastore or local folder path.
        :type path: typing.Union[azureml.data.data_reference.DataReference, str]
        :param in_memory: Whether to read the DataFrame from memory instead of persisting to disk.
        :type in_memory: bool
        :return: A Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().from_pandas_dataframe(dataframe, path, in_memory)

    @staticmethod
    @deprecated(
        '"Dataset.from_delimited_files"',
        '"Dataset.Tabular.from_delimited_files" instead')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_delimited_files(
            path,
            separator=',',
            header=PromoteHeadersBehavior.ALL_FILES_HAVE_SAME_HEADERS,
            encoding=FileEncoding.UTF8,
            quoting=False,
            infer_column_types=True,
            skip_rows=0,
            skip_mode=SkipLinesBehavior.NO_ROWS,
            comment=None,
            include_path=False,
            archive_options=None,
            partition_format=None):
        r"""Create an unregistered, in-memory Dataset from delimited files.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use ``Dataset.Tabular.from_delimited_files`` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Use this method to read delimited text files when you want to control the options used.

            After creating a Dataset, you should use :meth:`azureml.core.dataset.Dataset.get_profile` to list
            detected column types and summary statistics for each column.

            The returned Dataset is not registered with the workspace.

        .. code-block:: python

            # Create a dataset from delimited files with header option as ALL_FILES_HAVE_SAME_HEADERS
            dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'data/crime-spring.csv'),
                header='ALL_FILES_HAVE_SAME_HEADERS')

            df = dataset.to_pandas_dataframe()

        :param path: A data path in a registered datastore, a local path, or an HTTP URL.
        :type path: azureml.data.data_reference.DataReference or str
        :param separator: The separator used to split columns.
        :type separator: str
        :param header: Controls how column headers are promoted when reading from files.
        :type header: azureml.data.dataset_type_definitions.PromoteHeadersBehavior
        :param encoding: The encoding of the files being read.
        :type encoding: azureml.data.dataset_type_definitions.FileEncoding
        :param quoting: Specify how to handle new line characters within quotes.
            The default (False) is to interpret new line characters as starting new rows, irrespective of whether
            the new line characters are within quotes or not.
            If set to True, new line characters inside quotes will not result in new rows, and file reading
            speed will slow down.
        :type quoting: bool
        :param infer_column_types: Indicates whether column data types are inferred.
        :type infer_column_types: bool
        :param skip_rows: How many rows to skip in the file(s) being read.
        :type skip_rows: int
        :param skip_mode: Controls how rows are skipped when reading from files.
        :type skip_mode: azureml.data.dataset_type_definitions.SkipLinesBehavior
        :param comment: Character used to indicate comment lines in the files being read.
            Lines beginning with this string will be skipped.
        :type comment: str
        :param include_path: Whether to include a column containing the path of the file from which the data was read.
            This is useful when you are reading multiple files, and want to know which file a particular record
            originated from, or to keep useful information in file path.
        :type include_path: bool
        :param archive_options: Options for archive file, including archive type and entry glob pattern.
            We only support ZIP as archive type at the moment. For example, specifying

            .. code-block:: python

                archive_options = ArchiveOptions(archive_type = ArchiveType.ZIP, entry_glob = '*10-20.csv')

            reads all files with name ending with "10-20.csv" in ZIP.
        :type archive_options: azureml.dataprep.ArchiveOptions
        :param partition_format: Specify the partition format in path and create string columns from
            format '{x}' and datetime column from format '{x:yyyy/MM/dd/HH/mm/ss}', where 'yyyy', 'MM',
            'dd', 'HH', 'mm' and 'ss' are used to extrat year, month, day, hour, minute and second for the datetime
            type. The format should start from the postition of first partition key until the end of file path.
            For example, given a file path '../Accounts/2019/01/01/data.csv' where data is partitioned by
            department name and time, we can define '/{Department}/{PartitionDate:yyyy/MM/dd}/data.csv' to create
            columns 'Department' of string type and 'PartitionDate' of datetime type.
        :type partition_format: str
        :return: Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().from_delimited_files(
            path=path,
            separator=separator,
            header=header,
            encoding=encoding,
            quoting=quoting,
            infer_column_types=infer_column_types,
            skip_rows=skip_rows,
            skip_mode=skip_mode,
            comment=comment,
            include_path=include_path,
            archive_options=archive_options,
            partition_format=partition_format)

    @staticmethod
    @deprecated(
        '"Dataset.auto_read_files"',
        '"Dataset.Tabular.from_delimited_files" instead')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def auto_read_files(path, include_path=False, partition_format=None):
        r"""Analyzes the file(s) at the specified path and returns a new Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use the ``Dataset.Tabular.from_*`` methods to read files. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Use this method when to have file formats and delimiters detected automatically.

            After creating a Dataset, you should use :meth:`azureml.core.Dataset.get_profile` to list
            detected column types and summary statistics for each column.

            The returned Dataset is not registered with the workspace.


        :param path: A data path in a registered datastore, a local path, or an HTTP URL(CSV/TSV).
        :type path: azureml.data.data_reference.DataReference or str
        :param include_path: Whether to include a column containing the path of the file from which the data was read.
            Useful when reading multiple files, and want to know which file a particular record originated from.
            Also useful if there is information in file path or name that you want in a column.
        :type include_path: bool
        :param partition_format: Specify the partition format in path and create string columns from
            format '{x}' and datetime column from format '{x:yyyy/MM/dd/HH/mm/ss}', where 'yyyy', 'MM',
            'dd', 'HH', 'mm' and 'ss' are used to extrat year, month, day, hour, minute and second for the datetime
            type. The format should start from the postition of first partition key until the end of file path.
            For example, given a file path '../Accounts/2019/01/01/data.csv' where data is partitioned by
            department name and time, we can define '/{Department}/{PartitionDate:yyyy/MM/dd}/data.csv' to create
            columns 'Department' of string type and 'PartitionDate' of datetime type.
        :type partition_format: str
        :return: Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().auto_read_files(path, include_path, partition_format)

    @staticmethod
    @deprecated(
        '"Dataset.from_parquet_files"',
        '"Dataset.Tabular.from_parquet_files" instead')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_parquet_files(path, include_path=False, partition_format=None):
        r"""Create an unregistered, in-memory Dataset from parquet files.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use ``Dataset.Tabular.from_parquet_files`` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Use this method to read Parquet files.

            After creating a Dataset, you should use :meth:`azureml.core.dataset.Dataset.get_profile` to list
            detected column types and summary statistics for each column.

            The returned Dataset is not registered with the workspace.

        :param path: A data path in a registered datastore or a local path.
        :type path: azureml.data.data_reference.DataReference or str
        :param include_path: Whether to include a column containing the path of the file from which the data was read.
            This is useful when you are reading multiple files, and want to know which file a particular record
            originated from, or to keep useful information in file path.
        :type include_path: bool
        :param partition_format: Specify the partition format in path and create string columns from
            format '{x}' and datetime column from format '{x:yyyy/MM/dd/HH/mm/ss}', where 'yyyy', 'MM',
            'dd', 'HH', 'mm' and 'ss' are used to extrat year, month, day, hour, minute and second for the datetime
            type. The format should start from the postition of first partition key until the end of file path.
            For example, given a file path '../Accounts/2019/01/01/data.parquet' where data is partitioned by
            department name and time, we can define '/{Department}/{PartitionDate:yyyy/MM/dd}/data.parquet'
            to create columns 'Department' of string type and 'PartitionDate' of datetime type.
        :type partition_format: str
        :return: Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().from_parquet_files(path, include_path, partition_format)

    @staticmethod
    @deprecated('"Dataset.from_excel_files"')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_excel_files(
            path,
            sheet_name=None,
            use_column_headers=False,
            skip_rows=0,
            include_path=False,
            infer_column_types=True,
            partition_format=None):
        """Create an unregistered, in-memory Dataset from Excel files.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Use this method to read Excel files in .xlsx format. Data can be read from one sheet in each Excel file.
            After creating a Dataset, you should use :meth:`azureml.core.dataset.Dataset.get_profile` to list
            detected column types and summary statistics for each column. The returned Dataset is not registered
            with the workspace.

        :param path: A data path in a registered datastore or a local path.
        :type path: azureml.data.data_reference.DataReference or str
        :param sheet_name: The name of the Excel sheet to load.
            By default we read the first sheet from each Excel file.
        :type sheet_name: str
        :param use_column_headers: Controls whether to use the first row as column headers.
        :type use_column_headers: bool
        :param skip_rows: How many rows to skip in the file(s) being read.
        :type skip_rows: int
        :param include_path: Whether to include a column containing the path of the file from which the data was read.
            This is useful when you are reading multiple files, and want to know which file a particular record
            originated from, or to keep useful information in file path.
        :type include_path: bool
        :param infer_column_types: If true, column data types will be inferred.
        :type infer_column_types: bool
        :param partition_format: Specify the partition format in path and create string columns from
            format '{x}' and datetime column from format '{x:yyyy/MM/dd/HH/mm/ss}', where 'yyyy', 'MM',
            'dd', 'HH', 'mm' and 'ss' are used to extrat year, month, day, hour, minute and second for the datetime
            type. The format should start from the postition of first partition key until the end of file path.
            For example, given a file path '../Accounts/2019/01/01/data.xlsx' where data is partitioned by
            department name and time, we can define '/{Department}/{PartitionDate:yyyy/MM/dd}/data.xlsx'
            to create columns 'Department' of string type and 'PartitionDate' of datetime type.
        :type partition_format: str
        :return: Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().from_excel_files(
            path,
            sheet_name,
            use_column_headers,
            skip_rows,
            include_path,
            infer_column_types,
            partition_format)

    @staticmethod
    @deprecated(
        '"Dataset.from_binary_files"',
        '"Dataset.File.from_files" instead')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_binary_files(path):
        r"""Create an unregistered, in-memory Dataset from binary files.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use ``Dataset.File.from_files`` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Use this method to read files as streams of binary data. Returns one file stream object per
                file read. Use this method when you're reading images, videos, audio or other binary data.

            :meth:`azureml.core.dataset.Dataset.get_profile` and :meth:`azureml.core.dataset.Dataset.create_snapshot`
                will not work as expected for a Dataset created by this method.

            The returned Dataset is not registered with the workspace.

        :param path: A data path in a registered datastore or a local path.
        :type path: azureml.data.data_reference.DataReference or str
        :return: The Dataset object.
        :rtype: azureml.core.dataset.Dataset

        """
        return Dataset._client().from_binary_files(path)

    @staticmethod
    @deprecated(
        '"Dataset.from_sql_query"',
        '"Dataset.Tabular.from_sql_query" instead')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_sql_query(data_source, query):
        r"""Create an unregistered, in-memory Dataset from a SQL query.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use ``Dataset.Tabular.from_sql_query`` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        :param data_source: The details of the Azure SQL datastore.
        :type data_source: azureml.data.azure_sql_database_datastore.AzureSqlDatabaseDatastore
        :param query: The query to execute to read data.
        :type query: str
        :return: The local Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().from_sql_query(data_source, query)

    @staticmethod
    @deprecated(
        '"Dataset.from_json_files"',
        '"Dataset.Tabular.from_json_lines_files" to read from JSON lines file')
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_json_files(
        path,
        encoding=FileEncoding.UTF8,
        flatten_nested_arrays=False,
        include_path=False,
        partition_format=None
    ):
        r"""Create an unregistered, in-memory Dataset from JSON files.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use ``Dataset.Tabular.from_json_lines_files`` instead to read \
            from JSON lines file. For more information, see https://aka.ms/dataset-deprecation.

        :param path: The path to the file(s) or folder(s) that you want to load and parse. It can either be a
            local path or an Azure Blob url. Globbing is supported. For example, you can use path = "./data*" to
            read all files with name starting with "data".
        :type path: azureml.data.data_reference.DataReference or str
        :param encoding: The encoding of the files being read.
        :type encoding: azureml.data.dataset_type_definitions.FileEncoding
        :param flatten_nested_arrays: Property controlling program's handling of nested arrays.
            If you choose to flatten nested JSON arrays, it could result in a much larger number of rows.
        :type flatten_nested_arrays: bool
        :param include_path: Whether to include a column containing the path from which the data was read.
            This is useful when you are reading multiple files, and might want to know which file a particular record
            originated from, or to keep useful information in file path.
        :type include_path: bool
        :param partition_format: Specify the partition format in path and create string columns from
            format '{x}' and datetime column from format '{x:yyyy/MM/dd/HH/mm/ss}', where 'yyyy', 'MM',
            'dd', 'HH', 'mm' and 'ss' are used to extrat year, month, day, hour, minute and second for the datetime
            type. The format should start from the postition of first partition key until the end of file path.
            For example, given a file path '../Accounts/2019/01/01/data.json' and data is partitioned by
            department name and time, we can define '/{Department}/{PartitionDate:yyyy/MM/dd}/data.json'
            to create columns 'Department' of string type and 'PartitionDate' of datetime type.
        :type partition_format: str
        :return: The local Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().from_json_files(
            path=path,
            encoding=encoding,
            flatten_nested_arrays=flatten_nested_arrays,
            include_path=include_path,
            partition_format=partition_format)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def register(self, workspace, name, description=None, tags=None,
                 visible=True, exist_ok=False, update_if_exist=False):
        r"""Register the Dataset in the workspace, making it available to other users of the workspace.

        .. note::

            This method is deprecated and will no longer be supported.
            Recommend to use :meth:`azureml.data.abstract_dataset.AbstractDataset.register` instead. \
            For more information, see https://aka.ms/dataset-deprecation.

        :param workspace: The AzureML workspace in which the Dataset is to be registered.
        :type workspace: azureml.core.Workspace
        :param name: The name of the Dataset in the workspace.
        :type name: str
        :param description: A description of the Dataset.
        :type description: str
        :param tags: Tags to associate with the Dataset.
        :type tags: dict[str, str]
        :param visible: Indicates whether the Dataset is visible in the UI.
            If False, then the Dataset is hidden in the UI and available via SDK.
        :type visible: bool
        :param exist_ok: If True, the method returns the Dataset if it already exists in the given workspace,
            else error.
        :type exist_ok: bool
        :param update_if_exist: If ``exist_ok`` is True and ``update_if_exist`` is True, this method will update
            the definition and return the updated Dataset.
        :type update_if_exist: bool
        :return: A registered Dataset object in the workspace.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().register(
            workspace, name, self.definition, description, tags, visible, exist_ok, update_if_exist)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def update(self, name=None, description=None, tags=None, visible=None):
        """Update the Dataset mutable attributes in the workspace and return the updated Dataset from the workspace.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        :param name: The name of the Dataset in the workspace.
        :type name: str
        :param description: A description of the data.
        :type description: str
        :param tags: Tags to associate the Dataset with.
        :type tags: dict[str, str]
        :param visible: Indicates whether the the Dataset is visible in the UI.
        :type visible: bool
        :return: An updated Dataset object from the workspace.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().update(self.workspace, self.id, name, description, tags, visible)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def update_definition(self, definition, definition_update_message):
        """Update the Dataset definition.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            To consume the updated Dataset, use the object returned by this method.

        :param definition: The new definition of this Dataset.
        :type definition: azureml.data.dataset_definition.DatasetDefinition
        :param definition_update_message: The definition update message.
        :type definition_update_message: str
        :return: An updated Dataset object from the workspace.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().update_definition(self, definition, definition_update_message)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def to_pandas_dataframe(self):
        r"""Create a Pandas dataframe by executing the transformation pipeline defined by this Dataset definition.

        .. note::

            This method is deprecated and will no longer be supported.
            Create a \
            :class:`azureml.data.TabularDataset` by calling the static methods on \
            ``Dataset.Tabular`` and use the :meth:`azureml.data.TabularDataset.to_pandas_dataframe` \
            method there. For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Return a Pandas DataFrame fully materialized in memory.

        :return: A Pandas DataFrame.
        :rtype: pandas.DataFrame
        """
        return Dataset._client().to_pandas_dataframe(self.definition)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def to_spark_dataframe(self):
        r"""Create a Spark DataFrame that can execute the transformation pipeline defined by this Dataset definition.

        .. note::

            This method is deprecated and will no longer be supported.
            Create a \
            :class:`azureml.data.TabularDataset` by calling the static methods on ``Dataset.Tabular`` \
            and use the :meth:`azureml.data.TabularDataset.to_spark_dataframe` method there. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            The Spark Dataframe returned is only an execution plan and does not actually contain any data,
            as Spark Dataframes are lazily evaluated.

        :return: A Spark DataFrame.
        :rtype: pyspark.sql.DataFrame
        """
        return Dataset._client().to_spark_dataframe(self.definition)

    def head(self, count):
        """Pull the specified number of records specified from this Dataset and returns them as a DataFrame.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        :param count: The number of records to pull.
        :type count: int
        :return: A Pandas DataFrame.
        :rtype: pandas.DataFrame
        """
        return Dataset._client().head(self.definition, count)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def create_snapshot(self, snapshot_name, compute_target=None, create_data_snapshot=False, target_datastore=None):
        """Create a snapshot of the registered Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Snapshots capture point in time summary statistics of the underlying data
                and an optional copy of the data itself. To learn more about creating snapshots,
                go to https://aka.ms/azureml/howto/createsnapshots.


        :param snapshot_name: The snapshot name. Snapshot names should be unique within a Dataset.
        :type snapshot_name: str
        :param compute_target: Optional compute target to perform the snapshot profile creation.
            If omitted, the local compute is used.
        :type compute_target: typing.Union[azureml.core.compute.ComputeTarget, str]
        :param create_data_snapshot: If True, a materialized copy of the data will be created.
        :type create_data_snapshot: bool
        :param target_datastore: Target datastore to save snapshot.
            If omitted, the snapshot will be created in the default storage of the workspace.
        :type target_datastore: typing.Union[azureml.data.azure_storage_datastore.AbstractAzureStorageDatastore, str]
        :return: Dataset snapshot object.
        :rtype: azureml.data.dataset_snapshot.DatasetSnapshot
        """
        return Dataset._client().create_snapshot(self.definition, snapshot_name, compute_target,
                                                 create_data_snapshot, target_datastore)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_snapshot(self, snapshot_name):
        """Get snapshot of the Dataset by name.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        :param snapshot_name: The snapshot name.
        :type snapshot_name: str
        :return: Dataset snapshot object.
        :rtype: azureml.data.dataset_snapshot.DatasetSnapshot
        """
        return Dataset._client().get_snapshot(self.workspace, snapshot_name, self.id)

    def delete_snapshot(self, snapshot_name):
        """Delete snapshot of the Dataset by name.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::
            Use this to free up storage consumed by data saved in snapshots that you no longer need.

        :param snapshot_name: The snapshot name.
        :type snapshot_name: str
        :return: None.
        :rtype: None
        """
        return Dataset._client().delete_snapshot(self.workspace, snapshot_name, self.id)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_all_snapshots(self):
        """Get all snapshots of the Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        :return: List of Dataset snapshots.
        :rtype: builtin.list[azureml.data.dataset_snapshot.DatasetSnapshot]
        """
        return Dataset._client().get_all_snapshots(self.workspace, self.id)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def generate_profile(self, compute_target=None, workspace=None, arguments=None):
        """Generate new profile for the Dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

             Synchronous call, will block till it completes. Call
                :meth:`azureml.data.dataset_action_run.DatasetActionRun.get_result` to get the result of
                the action.

        :param compute_target: An optional compute target to perform the snapshot profile creation.
            If omitted, the local compute is used.
        :type compute_target: typing.Union[azureml.core.compute.ComputeTarget, str]
        :param workspace: Workspace, required for transient(unregistered) Datasets.
        :type workspace: azureml.core.Workspace
        :param arguments: Profile arguments. Valid arguments are:

            * 'include_stype_counts' of type bool. Check if values look like some well known semantic
              types such as email address, IP Address (V4/V6), US phone number, US zipcode,
              Latitude/Longitude. Enabling this impacts performance.

            * 'number_of_histogram_bins' of type int. Represents the number of histogram bins to use
              for numeric data. The default value is 10.

        :type arguments: dict[str, object]
        :return: Dataset action run object.
        :rtype: azureml.data.dataset_action_run.DatasetActionRun
        """
        return Dataset._client().generate_profile(self, compute_target, workspace, arguments)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def get_profile(self, arguments=None, generate_if_not_exist=True, workspace=None, compute_target=None):
        """Get summary statistics on the Dataset computed earlier.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            For a Dataset registered with an Azure Machine Learning workspace, this method retrieves an existing
                profile that was created earlier by calling ``get_profile`` if it is still valid. Profiles are
                invalidated when changed data is detected in the Dataset or the arguments to ``get_profile``
                are different from the ones used when the profile was generated. If the profile is not present
                or invalidated, ``generate_if_not_exist`` will determine if a new profile is generated.

            For a Dataset that is not registered with an Azure Machine Learning workspace, this method always
            runs :meth:`azureml.core.dataset.Dataset.generate_profile` and returns the result.

        :param arguments: Profile arguments.
        :type arguments: dict[str, object]
        :param generate_if_not_exist: Indicates whether to generate a profile if it doesn't exist.
        :type generate_if_not_exist: bool
        :param workspace: Workspace, required for transient(unregistered) Datasets.
        :type workspace: azureml.core.Workspace
        :param compute_target: A compute target to execute the profile action.
        :type compute_target: typing.Union[azureml.core.compute.ComputeTarget, str]
        :return: DataProfile of the Dataset.
        :rtype: azureml.dataprep.DataProfile
        """
        return Dataset._client().get_profile_with_state(
            self,
            arguments,
            generate_if_not_exist,
            workspace,
            compute_target)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def deprecate(self, deprecate_by_dataset_id):
        """Deprecate an active dataset in a workspace by another dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Deprecated Datasets will log warnings when they are consumed. Deprecating a dataset deprecates all
                its definitions.

            Deprecated Datasets can still be consumed. To completely block a Dataset from being consumed, archive it.

            If deprecated by accident, reactivate will activate it.

        :param deprecate_by_dataset_id: The Dataset ID which is the intended replacement for this Dataset.
        :type deprecate_by_dataset_id: str
        :return: None.
        :rtype: None
        """
        return Dataset._client().deprecate(self.workspace, self.id, self._etag, deprecate_by_dataset_id)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def archive(self):
        """Archive an active or deprecated dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            After archival, any attempt to consume the Dataset will result in an error.
            If archived by accident, reactivate will activate it.

        :return: None.
        :rtype: None
        """
        return Dataset._client().archive(self.workspace, self.id, self._etag)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def reactivate(self):
        """Reactivate an archived or deprecated dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        :return: None.
        :rtype: None
        """
        return Dataset._client().reactivate(self.workspace, self.id, self._etag)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def compare_profiles(self,
                         rhs_dataset,
                         profile_arguments=dict(),
                         include_columns=None,
                         exclude_columns=None,
                         histogram_compare_method=HistogramCompareMethod.WASSERSTEIN):
        """
        Compare the current Dataset's profile with another dataset profile.

        This shows the differences in summary statistics between two datasets. The parameter 'rhs_dataset'
        stands for "right-hand side", and is simply the second dataset. The first dataset
        (the current dataset object) is considered the "left-hand side".

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            This is for registered Datasets only.
            Raises an exception if the current Dataset's profile does not exist.
            For unregistered Datasets use profile.compare method.


        :param rhs_dataset: A second Dataset, also called a "right-hand side" Dataset for comparision.
        :type rhs_dataset: azureml.core.dataset.Dataset
        :param profile_arguments: Arguments to retrive specific profile.
        :type profile_arguments: dict
        :param include_columns: List of column names to be included in comparison.
        :type include_columns: builtin.list[str]
        :param exclude_columns: List of column names to be excluded in comparison.
        :type exclude_columns: builtin.list[str]
        :param histogram_compare_method: Enum describing the comparison method, ex: Wasserstein or Energy
        :type histogram_compare_method: azureml.data.dataset_type_definitions.HistogramCompareMethod
        :return: Difference between the two dataset profiles.
        :rtype: azureml.dataprep.api.engineapi.typedefinitions.DataProfileDifference
        """
        return Dataset._client().compare_dataset_profiles(
            lhs_dataset=self,
            rhs_dataset=rhs_dataset,
            profile_arguments=profile_arguments,
            compute_target=None,
            include_columns=include_columns,
            exclude_columns=exclude_columns,
            histogram_compare_method=histogram_compare_method)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def diff(self,
             rhs_dataset,
             compute_target=None,
             columns=None):
        """Diff the current Dataset with rhs_dataset.

        .. note::

            This method is deprecated and will no longer be supported.
            For more information, see https://aka.ms/dataset-deprecation.

        :param rhs_dataset: Another Dataset also called right hand side Dataset for comparision
        :type rhs_dataset: azureml.core.dataset.Dataset
        :param compute_target: compute target to perform the diff.
            If omitted, the local compute is used.
        :type compute_target: typing.Union[azureml.core.compute.ComputeTarget, str]
        :param columns: List of column names to be included in diff.
        :type columns: builtin.list[str]
        :return: Dataset action run object.
        :rtype: azureml.data.dataset_action_run.DatasetActionRun
        """
        return Dataset._client().compare_datasets(
            lhs_dataset=self,
            rhs_dataset=rhs_dataset,
            compute_target=compute_target,
            columns=columns)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def sample(self, sample_strategy, arguments):
        r"""Generate a new sample from the source Dataset, using the sampling strategy and parameters provided.

        .. note::

            This method is deprecated and will no longer be supported.
            Create a :class:`azureml.data.TabularDataset` by calling the static methods on \
            ``Dataset.Tabular`` and use the :meth:`azureml.data.TabularDataset.take_sample` method there. \
            For more information, see https://aka.ms/dataset-deprecation.

        .. remarks::

            Samples are generated by executing the transformation pipeline defined by this Dataset, and then
            applying the sampling strategy and parameters to the output data.  Each sampling method supports
            the following optional arguments:

            * top_n

                *  Optional arguments

                    * n, type integer. Select top N rows as your sample.

            * simple_random

                * Optional arguments

                    * probability, type float. Simple random sampling where each row has equal probability
                      of being selected. Probability should be a number between 0 and 1.

                    * seed, type float. Used by random number generator. Use for repeatability.

            * stratified

                * Optional arguments

                    * columns, type list[str]. List of strata columns in the data.

                    * seed, type float. Used by random number generator. Use for repeatability.

                    * fractions, type dict[tuple, float]. Tuple: column values that define a stratum,
                      must be in the same order as column names. Float: weight attached to a stratum
                      during sampling.

            The following code snippets are example design patterns for different sample methods.

            .. code-block:: python

                # sample_strategy "top_n"
                top_n_sample_dataset = dataset.sample('top_n', {'n': 5})

                # sample_strategy "simple_random"
                simple_random_sample_dataset = dataset.sample('simple_random', {'probability': 0.3, 'seed': 10.2})

                # sample_strategy "stratified"
                fractions = {}
                fractions[('THEFT',)] = 0.5
                fractions[('DECEPTIVE PRACTICE',)] = 0.2

                # take 50% of records with "Primary Type" as THEFT and 20% of records with "Primary Type" as
                # DECEPTIVE PRACTICE into sample Dataset
                sample_dataset = dataset.sample('stratified', {'columns': ['Primary Type'], 'fractions': fractions})


        :param sample_strategy: Sample strategy to use. Accepted values are "top_n", "simple_random", or "stratified".
        :type sample_strategy: str
        :param arguments: A dictionary with keys from the "Optional argument" in the
            list shown above, and values from tye "Type" column. Only arguments from the corresponding
            sampling method can be used. For example, for a "simple_random" sample type, you can only specify
            a dictionary with "probability" and "seed" keys.
        :type arguments: dict[str, object]
        :return: Dataset object as a sample of the original dataset.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().sample(self, sample_strategy, arguments, self._file_type())

    def _filter(self, expression):
        """
        Filter the data, leaving only the records that match the specified expression.

        .. remarks::

            Expressions are started by indexing the Dataset with the name of a column. They support a variety of
                functions and operators and can be combined using logical operators. The resulting expression will be
                lazily evaluated for each record when a data pull occurs and not where it is defined.

            .. code-block:: python
                from azureml.dataprep import col
                col('myColumn') > col('columnToCompareAgainst')
                col('myColumn').starts_with('prefix')

        :param expression: The expression to evaluate.
        :type expression: Expression
        :return: Filtered Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client().filter(self, expression, self._file_type())

    def _get_datapath(self):
        return self.definition._get_datapath()

    @staticmethod
    def _from_dataflow(
            dataflow):
        """Create unregistered, in-memory Dataset from dataflow.

        .. remarks::
            Use this method to convert dataflow to Dataset.

        :param dataflow: Dataflow object.
        :type dataflow: azureml.dataprep.Dataflow
        :return: Dataset object.
        :rtype: azureml.core.dataset.Dataset
        """
        return Dataset._client()._get_dataset_from_dataflow(
            dataflow=dataflow,
            file_data_path=Dataset._client()._get_source_path(dataflow),
            file_type=Dataset._client()._get_source_file_type(dataflow))

    @staticmethod
    def _get_definition_json(workspace, dataset_name, version=None):
        return Dataset._client()._get_definition_json(workspace, dataset_name, version)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def __str__(self):
        """Format Dataset data into a string.

        :return:
        :rtype: str
        """
        info = self._get_base_info_dict()
        formatted_info = ',\n'.join(["{}: {}".format(k, v) for k, v in info.items()])
        return "Dataset({0})".format(formatted_info)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def __repr__(self):
        """Representation of the object.

        :return: Return the string form of the Dataset object
        :rtype: str
        """
        return self.__str__()

    @track(_get_logger, activity_type=_PUBLIC_API)
    def __getitem__(self, key):
        r"""Keep the selected columns.

        .. note::

            This method is deprecated and will no longer be supported.
            Create a TabularDataset by \
            calling the static methods on ``Dataset.Tabular` and use the drop_columns/keep_columns method there. \
            For more information, see https://aka.ms/dataset-deprecation.

        Return a new dataflow always referencing the latest definition with an additional keep_columns
        transformation of the columns specific in the argument.

        :param key: The columns to keep.
        :type key: str or builtin.list
        :raises KeyError: if key is not of type str or list.
        :return: The new Dataset definition with the keep_columns transformation.
        :rtype: azureml.dataprep.Dataflow
        """
        if isinstance(key, str):
            return self.get_definition()[[key]]
        if isinstance(key, list):
            return self.get_definition()[key]
        raise KeyError('Only string or list of strings can be used to select columns')

    def _get_base_info_dict(self):
        """Return base info dictionary.

        :return:
        :rtype: OrderedDict
        """
        orderedDict = OrderedDict([
            ('Name', self.name),
            ('Workspace', self.workspace.name if self.workspace is not None else None),
            ('State', self.state),
            ('ID', self.id),
            ('Description', self.description),
            ('Tags', self.tags)
        ])
        return orderedDict

    def _get_base_info_dict_show(self):
        """Return base info dictionary.

        :return:
        :rtype: OrderedDict
        """
        orderedDict = OrderedDict([
            ('Name', self.name),
            ('Workspace', self.workspace.name if self.workspace is not None else None),
            ('State', self.state),
            ('ID', self.id),
            ('Description', self.description),
            ('Tags', self.tags),
            ('Latest version', self.definition_version),
            ('Created Time', self._created_time),
            ('Modified Time', self._modified_time),
            ('Datastore', self.definition._data_path.datastore_name),
            ('Relative Path', self.definition._data_path.relative_path)
        ])

        if self._deprecated_by_dataset_id is not None:
            orderedDict.update([
                ('Name', self.name),
                ('Workspace', self.workspace.name if self.workspace is not None else None),
                ('State', self.state),
                ('ID', self.id),
                ('Description', self.description),
                ('Tags', self.tags),
                ('Latest version', self.definition_version),
                ('Created Time', self._created_time),
                ('Modified Time', self._modified_time),
                ('Datastore', self.definition._data_path.datastore_name),
                ('Relative Path', self.definition._data_path.relative_path),
                ('Deprecated by dataset id', self._deprecated_by_dataset_id if
                 self._deprecated_by_dataset_id is not None else None)
            ])
        return orderedDict

    def _file_type(self):
        """Get the file type from the definition associated with Dataset.

        This is for SDK internal Use Only.
        :return: The file type.
        :rtype: str
        """
        return self.definition._file_type if self.definition is not None else None

    @staticmethod
    def _client():
        """Get Dataset client.

        :return: Returns the client
        :rtype: DatasetClient
        """
        from azureml.data._dataset_client import _DatasetClient
        return _DatasetClient

    class Scenario(object):
        """Defines constants for supported Dataset scenarios.

        These constants are used by Runs, Models, and Datadrift to identify what Datasets are
        used for in different parts of the Azure Machine Learning process.
        """

        TRAINING = "training"
        VALIDATION = "validation"
        INFERENCING = "inferencing"
