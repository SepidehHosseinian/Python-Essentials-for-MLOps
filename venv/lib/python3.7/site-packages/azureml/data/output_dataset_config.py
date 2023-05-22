# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains configurations that specifies how outputs for a job should be uploaded and promoted to a dataset.

For more information, see the article [how to specify
outputs](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets).
"""
from abc import abstractmethod
from copy import deepcopy
from uuid import uuid4

from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml.data._config_helper import _validate_name
from azureml.data._dataprep_helper import dataprep
from azureml.data._loggerfactory import _LoggerFactory, track
from azureml.data.constants import UPLOAD_MODE, MOUNT_MODE, DIRECT_MODE, HDFS_MODE, LINK_MODE, \
    _PUBLIC_API, _DATASET_OUTPUT_ARGUMENT_TEMPLATE
from azureml.data.dataset_factory import _set_column_types
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
from azureml.data.abstract_dataset import AbstractDataset

_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = _LoggerFactory.get_logger(__name__)
    return _logger


class OutputDatasetConfig:
    """Represent how to copy the output of a job to a remote storage location and be promoted to a Dataset.

    This is the base class used to represent how to copy the output of a job to a remote storage location, whether
    to register it as a named and versioned Dataset, and whether to apply any additional transformations to
    the Dataset that was created.

    You should not be creating instances of this class directly but instead should use the appropriate subclass.
    """

    @track(_get_logger, activity_type=_PUBLIC_API)
    def __init__(self, mode, name=None, **kwargs):
        """Initialize a OutputDatasetConfig.

        :param mode: The mode in which to copy the output to the remote storage.
        :type mode: str
        :param name: The name of the output specific to the run it will be produced in.
        :type name: str
        """
        self.mode = mode
        self._name = _validate_name(name, 'output') if name else OutputDatasetConfig._generate_random_name('output')
        self.arg_val = _DATASET_OUTPUT_ARGUMENT_TEMPLATE.format(self._name)
        self._additional_transformations = kwargs.get('additional_transformations')
        self._registration = kwargs.get('registration')
        self._origin = _Origin()

    @track(_get_logger, activity_type=_PUBLIC_API)
    def register_on_complete(self, name, description=None, tags=None):
        """Register the output as a new version of a named Dataset after the run has ran.

        If there are no datasets registered under the specified name, a new Dataset with the specified name will be
        registered. If there is a dataset registered under the specified name, then a new version will be added
        to this dataset.

        :param name: The Dataset name to register the output under.
        :type name: str
        :param description: The description for the Dataset.
        :type description: str
        :param tags: A list of tags to be assigned to the Dataset.
        :type tags: dict[str, str]
        :return: A new :class:`azureml.data.output_dataset_config.OutputDatasetConfig` instance with the registration
            information.
        :rtype: azureml.data.output_dataset_config.OutputDatasetConfig
        """
        if not name:
            raise ValueError('The registration name of the dataset must not be empty.')
        copy = deepcopy(self)
        copy._registration = RegistrationConfiguration(name, description, tags)
        return copy

    def as_input(self, name=None):
        """Specify how to consume the output as an input in subsequent pipeline steps.

        :param name: The name of the input specific to the run.
        :type name: str
        :return: A :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` instance describing
            how to deliver the input data.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        raise NotImplementedError()

    @property
    def name(self):
        """Name of the output.

        :return: Name of the output.
        """
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the output."""
        self._name = _validate_name(value, 'output')

    @staticmethod
    def _generate_random_name(prefix):
        return '{}_{}'.format(prefix, str(uuid4()).split('-')[0])

    def _to_output_data(self):
        from azureml.core.runconfig import OutputData, RegistrationOptions, OutputOptions, DatasetRegistrationOptions

        def to_registration(registration):
            dataflow_json = self._additional_transformations.to_json() if self._additional_transformations else None
            dataset_registration_options = DatasetRegistrationOptions(dataflow_json)
            if not registration:
                return RegistrationOptions(dataset_registration_options=dataset_registration_options)
            return RegistrationOptions(registration.name, registration.description, registration.tags,
                                       dataset_registration_options=dataset_registration_options)

        registration = to_registration(self._registration)
        output_options = OutputOptions(registration_options=registration)
        return OutputData(mechanism=self.mode, additional_options=output_options)

    def _set_producer(self, step):
        """Pipeline specific method."""
        self._origin.producer_step = step

    @property
    def _producer(self):
        """Pipeline specific property."""
        return self._origin.producer_step


class TransformationMixin:
    """This class provides transformation capabilities to output datasets."""

    @track(_get_logger, custom_dimensions={'app_name': 'TransformationMixin'}, activity_type=_PUBLIC_API)
    def read_delimited_files(
            self, include_path=False, separator=',', header=PromoteHeadersBehavior.ALL_FILES_HAVE_SAME_HEADERS,
            partition_format=None, path_glob=None, set_column_types=None):
        """Transform the output dataset to a tabular dataset by reading all the output as delimited files.

        :param include_path: Boolean to keep path information as column in the dataset. Defaults to False.
            This is useful when reading multiple files, and want to know which file a particular record
            originated from, or to keep useful information in file path.
        :type include_path: bool
        :param separator: The separator used to split columns.
        :type separator: str
        :param header: Controls how column headers are promoted when reading from files. Defaults to assume
            that all files have the same header.
        :type header: azureml.data.dataset_type_definitions.PromoteHeadersBehavior
        :param partition_format: Specify the partition format of path. Defaults to None.
            The partition information of each path will be extracted into columns based on the specified format.
            Format part '{column_name}' creates string column, and '{column_name:yyyy/MM/dd/HH/mm/ss}' creates
            datetime column, where 'yyyy', 'MM', 'dd', 'HH', 'mm' and 'ss' are used to extract year, month, day,
            hour, minute and second for the datetime type. The format should start from the position of first
            partition key until the end of file path.
            For example, given the path '../Accounts/2019/01/01/data.parquet' where the partition is by
            department name and time, partition_format='/{Department}/{PartitionDate:yyyy/MM/dd}/data.parquet'
            creates a string column 'Department' with the value 'Accounts' and a datetime column 'PartitionDate'
            with the value '2019-01-01'.
        :type partition_format: str
        :param path_glob: A glob-like pattern to filter files that will be read as delimited files. If set to None,
            then all files will be read as delimited files.

            Glob is a Unix style pathname pattern expansion: https://docs.python.org/3/library/glob.html

            ex)
            - `*.csv` -> selects files with `.csv` file extension
            - `test_*.csv` -> selects files with filenames that startwith `test_` and has `.csv` file extension
            - `/myrootdir/project_one/*/*/*.txt` -> selects files that are two subdirectories deep in
            `/myrootdir/project_one/` and have `.txt` file extension

            Note: Using the `**` pattern in large directory trees may consume an inordinate amount of time.
            In general, for large directory trees, being more specific in the glob pattern can increase performance.
        :type path_glob: str
        :param set_column_types: A dictionary to set column data type, where key is column name and value is
            :class:`azureml.data.DataType`. Columns not in the dictionary will remain of type string. Passing None
            will result in no conversions. Entries for columns not found in the source data will not cause an error
            and will be ignored.
        :type set_column_types: dict[str, azureml.data.DataType]
        :return: A :class:`azureml.data.output_dataset_config.OutputTabularDatasetConfig` instance with instruction of
            how to convert the output into a TabularDataset.
        :rtype: azureml.data.output_dataset_config.OutputTabularDatasetConfig
        """
        dprep = dataprep()
        dataflow = dprep.Dataflow(self._engine_api)
        dataflow = TransformationMixin._filter_path(dprep, dataflow, path_glob)
        dataflow = dataflow.parse_delimited(
            separator=separator, headers_mode=header, encoding=dprep.FileEncoding.UTF8, quoting=False,
            skip_rows=0, skip_mode=dprep.SkipMode.NONE, comment=None
        )
        if partition_format:
            dataflow = dataflow._add_columns_from_partition_format('Path', partition_format, False)
        dataflow = TransformationMixin._handle_path(dataflow, include_path)
        dataflow = _set_column_types(dataflow, set_column_types)
        return self._from_dataflow(dataflow)

    @track(_get_logger, custom_dimensions={'app_name': 'TransformationMixin'}, activity_type=_PUBLIC_API)
    def read_parquet_files(self, include_path=False, partition_format=None, path_glob=None,
                           set_column_types=None):
        """Transform the output dataset to a tabular dataset by reading all the output as Parquet files.

        The tabular dataset is created by parsing the parquet file(s) pointed to by the intermediate output.

        :param include_path: Boolean to keep path information as column in the dataset. Defaults to False.
            This is useful when reading multiple files, and want to know which file a particular record
            originated from, or to keep useful information in file path.
        :type include_path: bool
        :param partition_format: Specify the partition format of path. Defaults to None.
            The partition information of each path will be extracted into columns based on the specified format.
            Format part '{column_name}' creates string column, and '{column_name:yyyy/MM/dd/HH/mm/ss}' creates
            datetime column, where 'yyyy', 'MM', 'dd', 'HH', 'mm' and 'ss' are used to extract year, month, day,
            hour, minute and second for the datetime type. The format should start from the position of first
            partition key until the end of file path.
            For example, given the path '../Accounts/2019/01/01/data.parquet' where the partition is by
            department name and time, partition_format='/{Department}/{PartitionDate:yyyy/MM/dd}/data.parquet'
            creates a string column 'Department' with the value 'Accounts' and a datetime column 'PartitionDate'
            with the value '2019-01-01'.
        :type partition_format: str
        :param path_glob: A glob-like pattern to filter files that will be read as parquet files. If set to None,
            then all files will be read as parquet files.

            Glob is a Unix style pathname pattern expansion: https://docs.python.org/3/library/glob.html

            ex)
            - `*.parquet` -> selects files with `.parquet` file extension
            - `test_*.parquet` -> selects files with filenames that startwith `test_`
            and has `.parquet` file extension
            - `/myrootdir/project_one/*/*/*.parquet` -> selects files that are two subdirectories deep in
            `/myrootdir/project_one/` and have `.parquet` file extension

            Note: Using the `**` pattern in large directory trees may consume an inordinate amount of time.
            In general, for large directory trees, being more specific in the glob pattern can increase performance.
        :type path_glob: str
        :param set_column_types: A dictionary to set column data type, where key is column name and value is
            :class:`azureml.data.DataType`. Columns not in the dictionary will remain of type loaded from the parquet
            file. Passing None will result in no conversions. Entries for columns not found in the source data will
            not cause an error and will be ignored.
        :type set_column_types: dict[str, azureml.data.DataType]
        :return: A :class:`azureml.data.output_dataset_config.OutputTabularDatasetConfig` instance with instruction of
            how to convert the output into a TabularDataset.
        :rtype: azureml.data.output_dataset_config.OutputTabularDatasetConfig
        """
        dprep = dataprep()
        dataflow = dprep.Dataflow(self._engine_api)
        dataflow = TransformationMixin._filter_path(dprep, dataflow, path_glob)
        dataflow = dataflow.read_parquet_file()
        if partition_format:
            dataflow = dataflow._add_columns_from_partition_format('Path', partition_format, False)
        dataflow = TransformationMixin._handle_path(dataflow, include_path)
        dataflow = _set_column_types(dataflow, set_column_types)
        return self._from_dataflow(dataflow)

    @staticmethod
    def _filter_path(dprep, dataflow, path_glob):
        if not path_glob:
            return dataflow

        return dataflow.filter(
            dprep.RegEx(path_glob).is_match(dprep.api.functions.get_stream_name(dataflow['Path']))
        )

    @staticmethod
    def _handle_path(dataflow, include_path):
        if not include_path:
            return dataflow.drop_columns('Path')
        return dataflow

    @property
    def _engine_api(self):
        return dataprep().api.engineapi.api.get_engine_api()

    @abstractmethod
    def _from_dataflow(self, dataflow) -> AbstractDataset:
        pass


class OutputFileDatasetConfig(OutputDatasetConfig, TransformationMixin):
    """Represent how to copy the output of a run and be promoted as a FileDataset.

    The OutputFileDatasetConfig allows you to specify how you want a particular local path on the compute target
    to be uploaded to the specified destination. If no arguments are passed to the constructor, we will
    automatically generate a name, a destination, and a local path.

    An example of not passing any arguments:

    .. code-block:: python

        workspace = Workspace.from_config()
        experiment = Experiment(workspace, 'output_example')

        output = OutputFileDatasetConfig()

        script_run_config = ScriptRunConfig('.', 'train.py', arguments=[output])

        run = experiment.submit(script_run_config)
        print(run)

    An example of creating an output then promoting the output to a tabular dataset and register it with
    name foo:

    .. code-block:: python

        workspace = Workspace.from_config()
        experiment = Experiment(workspace, 'output_example')

        datastore = Datastore(workspace, 'example_adls_gen2_datastore')

        # for more information on the parameters and methods, please look for the corresponding documentation.
        output = OutputFileDatasetConfig().read_delimited_files().register_on_complete('foo')

        script_run_config = ScriptRunConfig('.', 'train.py', arguments=[output])

        run = experiment.submit(script_run_config)
        print(run)

    .. remarks::
        You can pass the OutputFileDatasetConfig as an argument to your run, and it will be automatically
        translated into local path on the compute. The source argument will be used if one is specified,
        otherwise we will automatically generate a directory in the OS's temp folder. The files and folders inside
        the source directory will then be copied to the destination based on the output configuration.

        By default the mode by which the output will be copied to the destination storage will be set to mount.
        For more information about mount mode, please see the documentation for as_mount.

    :param name: The name of the output specific to this run. This is generally used for lineage purposes. If set
        to None, we will automatically generate a name. The name will also become an environment variable which
        contains the local path of where you can write your output files and folders to that will be uploaded to
        the destination.
    :type name: str
    :param destination: The destination to copy the output to. If set to None, we will copy the output to the
        workspaceblobstore datastore, under the path /dataset/{run-id}/{output-name}, where `run-id` is the Run's
        ID and the `output-name` is the output name from the `name` parameter above. The destination is a tuple
        where the first item is the datastore and the second item is the path within the datastore to copy the
        data to.

        The path within the datastore can be a template path. A template path is just a regular path but with
        placeholders inside. Those placeholders will then be resolved at the appropriate time. The syntax for
        placeholders is {placeholder}, for example, /path/with/{placeholder}. Currently only two placeholders
        are supported, {run-id} and {output-name}.
    :type destination: tuple
    :param source: The path within the compute target to copy the data from. If set to None, we
        will set this to a directory we create inside the compute target's OS temporary directory.
    :type source: str
    :param partition_format: Specify the partition format of path. Defaults to None.
            The partition information of each path will be extracted into columns based on the specified format.
            Format part '{column_name}' creates string column, and '{column_name:yyyy/MM/dd/HH/mm/ss}' creates
            datetime column, where 'yyyy', 'MM', 'dd', 'HH', 'mm' and 'ss' are used to extract year, month, day,
            hour, minute and second for the datetime type. The format should start from the position of first
            partition key until the end of file path.
            For example, given the path '../Accounts/2019/01/01/data.parquet' where the partition is by
            department name and time, partition_format='/{Department}/{PartitionDate:yyyy/MM/dd}/data.parquet'
            creates a string column 'Department' with the value 'Accounts' and a datetime column 'PartitionDate'
            with the value '2019-01-01'.
    :type partition_format: str
    """

    @track(_get_logger, custom_dimensions={'app_name': 'OutputFileDatasetConfig'}, activity_type=_PUBLIC_API)
    def __init__(self, name=None, destination=None, source=None, partition_format=None):
        """Initialize a OutputFileDatasetConfig.

        The OutputFileDatasetConfig allows you to specify how you want a particular local path on the compute target
        to be uploaded to the specified destination. If no arguments are passed to the constructor, we will
        automatically generate a name, a destination, and a local path.

        An example of not passing any arguments:

        .. code-block:: python

            workspace = Workspace.from_config()
            experiment = Experiment(workspace, 'output_example')

            output = OutputFileDatasetConfig()

            script_run_config = ScriptRunConfig('.', 'train.py', arguments=[output])

            run = experiment.submit(script_run_config)
            print(run)

        An example of creating an output then promoting the output to a tabular dataset and register it with
        name foo:

        .. code-block:: python

            workspace = Workspace.from_config()
            experiment = Experiment(workspace, 'output_example')

            datastore = Datastore(workspace, 'example_adls_gen2_datastore')

            # for more information on the parameters and methods, please look for the corresponding documentation.
            output = OutputFileDatasetConfig().read_delimited_files().register_on_complete('foo')

            script_run_config = ScriptRunConfig('.', 'train.py', arguments=[output])

            run = experiment.submit(script_run_config)
            print(run)

        .. remarks::
            You can pass the OutputFileDatasetConfig as an argument to your run, and it will be automatically
            translated into local path on the compute. The source argument will be used if one is specified,
            otherwise we will automatically generate a directory in the OS's temp folder. The files and folders inside
            the source directory will then be copied to the destination based on the output configuration.

            By default the mode by which the output will be copied to the destination storage will be set to mount.
            For more information about mount mode, please see the documentation for as_mount.

        :param name: The name of the output specific to this run. This is generally used for lineage purposes. If set
            to None, we will automatically generate a name. The name will also become an environment variable which
            contains the local path of where you can write your output files and folders to that will be uploaded to
            the destination.
        :type name: str
        :param destination: The destination to copy the output to. If set to None, we will copy the output to the
            workspaceblobstore datastore, under the path /dataset/{run-id}/{output-name}, where `run-id` is the Run's
            ID and the `output-name` is the output name from the `name` parameter above. The destination is a tuple
            where the first item is the datastore and the second item is the path within the datastore to copy the
            data to.

            The path within the datastore can be a template path. A template path is just a regular path but with
            placeholders inside. Those placeholders will then be resolved at the appropriate time. The syntax for
            placeholders is {placeholder}, for example, /path/with/{placeholder}. Currently only two placeholders
            are supported, {run-id} and {output-name}.
        :type destination: tuple
        :param source: The path within the compute target to copy the data from. If set to None, we
            will set this to a directory we create inside the compute target's OS temporary directory.
        :type source: str
        :param partition_format: Specify the partition format of path. Defaults to None.
            The partition information of each path will be extracted into columns based on the specified format.
            Format part '{column_name}' creates string column, and '{column_name:yyyy/MM/dd/HH/mm/ss}' creates
            datetime column, where 'yyyy', 'MM', 'dd', 'HH', 'mm' and 'ss' are used to extract year, month, day,
            hour, minute and second for the datetime type. The format should start from the position of first
            partition key until the end of file path.
            For example, given the path '../Accounts/2019/01/01/data.parquet' where the partition is by
            department name and time, partition_format='/{Department}/{PartitionDate:yyyy/MM/dd}/data.parquet'
            creates a string column 'Department' with the value 'Accounts' and a datetime column 'PartitionDate'
            with the value '2019-01-01'.
        :type partition_format: str
        """
        super(OutputFileDatasetConfig, self).__init__(MOUNT_MODE, name)
        self.destination = destination
        self.source = source
        self._upload_options = None
        self._mount_options = None
        if partition_format:
            dprep = dataprep()
            dataflow = dprep.Dataflow(self._engine_api)
            dataflow = dataflow._add_columns_from_partition_format('Path', partition_format, False)
            self._additional_transformations = dataflow

    @track(_get_logger, custom_dimensions={'app_name': 'OutputFileDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_mount(self, disable_metadata_cache=False):
        """Set the mode of the output to mount.

        For mount mode, the output directory will be a FUSE mounted directory. Files written to the mounted directory
        will be uploaded when the file is closed.

        :param disable_metadata_cache: Whether to cache metadata in local node,
            if disabled a node will not be able to see files generated from other nodes during job running.
        :type disable_metadata_cache: bool
        :return: A :class:`azureml.data.OutputFileDatasetConfig` instance with mode set to mount.
        :rtype: azureml.data.OutputFileDatasetConfig
        """
        copy = deepcopy(self)
        copy.mode = MOUNT_MODE
        copy._mount_options = MountOptions(disable_metadata_cache)
        copy._upload_options = None
        return copy

    @track(_get_logger, custom_dimensions={'app_name': 'OutputFileDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_upload(self, overwrite=False, source_globs=None):
        """Set the mode of the output to upload.

        For upload mode, files written to the output directory will be uploaded at the end of the job. If the job
        fails or gets canceled, then the output directory will not be uploaded.

        :param overwrite: Whether to overwrite files that already exists in the destination.
        :type overwrite: bool
        :param source_globs: Glob patterns used to filter files that will be uploaded.
        :type source_globs: builtin.list[str]
        :return: A :class:`azureml.data.OutputFileDatasetConfig` instance with mode set to upload.
        :rtype: azureml.data.OutputFileDatasetConfig
        """
        copy = deepcopy(self)
        copy.mode = UPLOAD_MODE
        copy._mount_options = None
        copy._upload_options = UploadOptions(overwrite, source_globs)
        return copy

    @track(_get_logger, custom_dimensions={'app_name': 'OutputFileDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_input(self, name=None):
        """Specify how to consume the output as an input in subsequent pipeline steps.

        :param name: The name of the input specific to the run.
        :type name: str
        :return: A :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` instance describing
            how to deliver the input data.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

        name = name or self.__class__._generate_random_name('input')
        return DatasetConsumptionConfig(name, self, 'mount')

    def _to_output_data(self):
        from azureml.core.runconfig import DataLocation, DataPath, UploadOptions as RunConfigUploadOptions,\
            MountOptions as RunConfigMountOptions, GlobsOptions

        def to_data_location(destination):
            if not destination:
                return None
            try:
                datastore_name = destination[0].name
            except AttributeError:
                datastore_name = destination[0]
            return DataLocation(data_path=DataPath(datastore_name, *destination[1:]))

        def to_upload_options(mode, upload_options):
            if mode != UPLOAD_MODE:
                if upload_options:
                    raise ValueError('Mode is not set to upload but an UploadOption is provided. Please make sure the '
                                     'mode is set to upload by calling as_upload.')
                return None
            if upload_options:
                globs = to_globs(upload_options.source_globs)
                return RunConfigUploadOptions(upload_options.overwrite, globs)
            return RunConfigUploadOptions(False, None)

        def to_mount_options(mode, mount_options):
            if mode != MOUNT_MODE:
                if mount_options:
                    raise ValueError('Mode is not set to mount but an MountOption is provided. Please make sure the '
                                     'mode is set to mount by calling as_mount.')
                return None
            if mount_options:
                return RunConfigMountOptions(mount_options.disable_metadata_cache)
            return RunConfigMountOptions()

        def to_globs(globs):
            if not globs:
                return None
            if not isinstance(globs, list):
                globs = [globs]
            return GlobsOptions(globs)

        output_data = super(OutputFileDatasetConfig, self)._to_output_data()
        output_data.output_location = to_data_location(self.destination)
        output_data.additional_options.upload_options = to_upload_options(self.mode, self._upload_options)
        output_data.additional_options.mount_options = to_mount_options(self.mode, self._mount_options)
        output_data.additional_options.path_on_compute = self.source
        return output_data

    def _from_dataflow(self, dataflow) -> AbstractDataset:
        return _create_tabular_dataset_with_dataflow(self, dataflow)


class HDFSOutputDatasetConfig(OutputDatasetConfig, TransformationMixin):
    """Represent how to output to a HDFS path and be promoted as a FileDataset."""

    @track(_get_logger, custom_dimensions={'app_name': 'HDFSOutputDatasetConfig'}, activity_type=_PUBLIC_API)
    def __init__(self, name=None, destination=None):
        """Initialize a HDFSOutputDatasetConfig.

        .. remarks::
            You can pass the HDFSOutputDatasetConfig as an argument of a run and it will be automatically translated
            to HDFS path.

        :param name: The name of the output specific to this run. This is generally used for lineage purposes. If set
            to None, we will automatically generate a name.
        :type name: str
        :param destination: The destination of the output. If set to None, it will be outputted to workspaceblobstore
            datastore, under the path /dataset/{run-id}/{output-name}, where `run-id` is the Run's ID and the
            `output-name` is the output name from the `name` parameter above. The destination is a tuple where the
            first item is the datastore and the second item is the path within the datastore.
        :type destination: tuple
        """
        super(HDFSOutputDatasetConfig, self).__init__(HDFS_MODE, name)
        self.destination = destination
        # TODO: This is a workaround, find a correct way to fix the source and upload options
        self.source = None
        self._upload_options = None

    def _to_output_data(self):
        from azureml.core.runconfig import DataLocation, DataPath

        def to_data_location(destination):
            if not destination:
                return None
            try:
                datastore_name = destination[0].name
            except AttributeError:
                datastore_name = destination[0]
            return DataLocation(data_path=DataPath(datastore_name, *destination[1:]))

        output_data = super(HDFSOutputDatasetConfig, self)._to_output_data()
        output_data.output_location = to_data_location(self.destination)
        return output_data

    @track(_get_logger, custom_dimensions={'app_name': 'HDFSOutputDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_input(self, name=None):
        """Specify how to consume the output as an input in subsequent pipeline steps.

        :param name: The name of the input specific to the run.
        :type name: str
        :return: A :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` instance describing
            how to deliver the input data.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

        name = name or self.__class__._generate_random_name('input')
        return DatasetConsumptionConfig(name, self, MOUNT_MODE)

    def _from_dataflow(self, dataflow) -> AbstractDataset:
        return _create_tabular_dataset_with_dataflow(self, dataflow)


class OutputTabularDatasetConfig(OutputDatasetConfig):
    """Represent how to copy the output of a run and be promoted as a TabularDataset.

    .. remarks::

        You should not call this constructor directly, but instead should create a OutputFileDatasetConfig and then
        call the corresponding read_* methods to convert it into a OutputTabularDatasetConfig.

        The way the output will be copied to the destination for a OutputTabularDatasetConfig is the same as a
        OutputFileDatasetConfig. The difference between them is that the Dataset that is created will be a
        TabularDataset containing all the specified transformations.
    """

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def __init__(self, **kwargs):
        """Initialize a OutputTabularDatasetConfig.

        .. remarks::

            You should not call this constructor directly, but instead should create a OutputFileDatasetConfig and then
            call the corresponding read_* methods to convert it into a OutputTabularDatasetConfig.

            The way the output will be copied to the destination for a OutputTabularDatasetConfig is the same as a
            OutputFileDatasetConfig. The difference between them is that the Dataset that is created will be a
            TabularDataset containing all the specified transformations.

        """
        output_file_dataset_config = kwargs.get('output_file_dataset_config')

        if not output_file_dataset_config:
            raise ValueError('The constructor of this class is not supposed to be called directly. Please create '
                             'an OutputFileDatasetConfig and add additional transforms to get a '
                             'OutputTabularDatasetConfig.')

        super(OutputTabularDatasetConfig, self).__init__(
            output_file_dataset_config.mode,
            output_file_dataset_config.name,
            additional_transformations=output_file_dataset_config._additional_transformations
        )
        self._output_file_dataset_config = output_file_dataset_config
        self._origin = output_file_dataset_config._origin

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def keep_columns(self, columns):
        """Keep the specified columns and drops all others from the Dataset.

        :param columns: The name or a list of names for the columns to keep.
        :type columns: typing.Union[str, builtin.list[str]]
        :return: A :class:`azureml.data.output_dataset_config.OutputTabularDatasetConfig` instance with which columns
            to keep.
        :rtype: azureml.pipeline.core.pipeline_output_dataset.PipelineOutputTabularDataset
        """
        dataflow = self._additional_transformations.keep_columns(columns)
        return _create_tabular_dataset_with_dataflow(self._output_file_dataset_config, dataflow)

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def drop_columns(self, columns):
        """Drop the specified columns from the Dataset.

        :param columns: The name or a list of names for the columns to drop.
        :type columns: typing.Union[str, builtin.list[str]]
        :return: A :class:`azureml.data.output_dataset_config.OutputTabularDatasetConfig` instance with which columns
            to drop.
        :rtype: azureml.pipeline.core.pipeline_output_dataset.PipelineOutputTabularDataset
        """
        dataflow = self._additional_transformations.drop_columns(columns)
        return _create_tabular_dataset_with_dataflow(self._output_file_dataset_config, dataflow)

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def random_split(self, percentage, seed=None):
        """Split records in the dataset into two parts randomly and approximately by the percentage specified.

        The resultant output configs will have their names changed, the first one will have _1 appended to the name
        and the second one will have _2 appended to the name. If it will cause a name collision or you would like to
        specify a custom name, please manually set their names.

        :param percentage: The approximate percentage to split the dataset by. This must be a number between
            0.0 and 1.0.
        :type percentage: float
        :param seed: Optional seed to use for the random generator.
        :type seed: int
        :return: Returns a tuple of two OutputTabularDatasetConfig objects representing the two Datasets after the
            split.
        :rtype: tuple(azureml.data.output_dataset_config.OutputTabularDatasetConfig,
            azureml.data.output_dataset_config.OutputTabularDatasetConfig)
        """
        dataflow1, dataflow2 = self._additional_transformations.random_split(percentage, seed)
        first = _create_tabular_dataset_with_dataflow(self._output_file_dataset_config, dataflow1)
        second = _create_tabular_dataset_with_dataflow(self._output_file_dataset_config, dataflow2)
        first.name += '_1'
        second.name += '_2'
        return first, second

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_mount(self):
        """Set the mode of the output to mount.

        For mount mode, the output directory will be a FUSE mounted directory. Files written to the mounted directory
        will be uploaded when the file is closed.

        :return: A :class:`azureml.data.output_dataset_config.OutputTabularDatasetConfig` instance with mode set to
            mount.
        :rtype: azureml.data.output_dataset_config.OutputTabularDatasetConfig
        """
        copy = deepcopy(self)
        copy.mode = MOUNT_MODE
        copy._output_file_dataset_config.mode = MOUNT_MODE
        copy._output_file_dataset_config._upload_options = None
        return copy

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_upload(self, overwrite=False, source_globs=None):
        """Set the mode of the output to upload.

        For upload mode, files written to the output directory will be uploaded at the end of the job. If the job
        fails or gets canceled, then the output directory will not be uploaded.

        :param overwrite: Whether to overwrite files that already exists in the destination.
        :type overwrite: bool
        :param source_globs: Glob patterns used to filter files that will be uploaded.
        :type source_globs: builtin.list[str]
        :return: A :class:`azureml.data.output_dataset_config.OutputTabularDatasetConfig` instance with mode set to
            upload.
        :rtype: azureml.data.output_dataset_config.OutputTabularDatasetConfig
        """
        copy = deepcopy(self)
        copy.mode = UPLOAD_MODE
        copy._output_file_dataset_config.mode = UPLOAD_MODE
        copy._output_file_dataset_config._upload_options = UploadOptions(overwrite, source_globs)
        return copy

    @track(_get_logger, custom_dimensions={'app_name': 'OutputTabularDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_input(self, name=None):
        """Specify how to consume the output as an input in subsequent pipeline steps.

        :param name: The name of the input specific to the run.
        :type name: str
        :return: A :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` instance describing
            how to deliver the input data.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

        name = name or self.__class__._generate_random_name('input')
        return DatasetConsumptionConfig(name, self, DIRECT_MODE)

    def _to_output_data(self):
        return self._output_file_dataset_config._to_output_data()


@experimental
class LinkFileOutputDatasetConfig(OutputDatasetConfig):
    """Represent how to link the output of a run and be promoted as a FileDataset.

    The LinkFileOutputDatasetConfig allows you to link a file dataset as output dataset

    .. code-block:: python

        workspace = Workspace.from_config()
        experiment = Experiment(workspace, 'output_example')

        output = LinkFileOutputDatasetConfig('link_output')

        script_run_config = ScriptRunConfig('.', 'link.py', arguments=[output])

        # within link.py
        # from azureml.core import Run, Dataset
        # run = Run.get_context()
        # workspace = run.experiment.workspace
        # dataset = Dataset.get_by_name(workspace, name='dataset_to_link')
        # run.output_datasets['link_output'].link(dataset)

        run = experiment.submit(script_run_config)
        print(run)

    """

    @track(_get_logger, custom_dimensions={'app_name': 'LinkFileOutputDatasetConfig'}, activity_type=_PUBLIC_API)
    def __init__(self, name=None):
        """Initialize a LinkFileOutputDatasetConfig.

        :param name: The name of the output specific to this run.
        :type name: str
        """
        super(LinkFileOutputDatasetConfig, self).__init__(LINK_MODE, name)

    @track(_get_logger, custom_dimensions={'app_name': 'LinkFileOutputDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_input(self, name=None):
        """Specify how to consume the output as an input in subsequent pipeline steps.

        :param name: The name of the input specific to the run.
        :type name: str
        :return: A :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` instance describing
            how to deliver the input data.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

        name = name or self.__class__._generate_random_name('input')
        return DatasetConsumptionConfig(name, self, MOUNT_MODE)


@experimental
class LinkTabularOutputDatasetConfig(OutputDatasetConfig):
    """Represent how to link the output of a run and be promoted as a TabularDataset.

    The LinkTabularOutputDatasetConfig allows you to link a file Tabular as output dataset

    .. code-block:: python

        workspace = Workspace.from_config()
        experiment = Experiment(workspace, 'output_example')

        output = LinkTabularOutputDatasetConfig('link_output')

        script_run_config = ScriptRunConfig('.', 'link.py', arguments=[output])

        # within link.py
        # from azureml.core import Run, Dataset
        # run = Run.get_context()
        # workspace = run.experiment.workspace
        # dataset = Dataset.get_by_name(workspace, name='dataset_to_link')
        # run.output_datasets['link_output'].link(dataset)

        run = experiment.submit(script_run_config)
        print(run)

    """

    @track(_get_logger, custom_dimensions={'app_name': 'LinkTabularOutputDatasetConfig'}, activity_type=_PUBLIC_API)
    def __init__(self, name=None):
        """Initialize a LinkTabularOutputDatasetConfig.

        :param name: The name of the output specific to this run.
        :type name: str
        """
        super(LinkTabularOutputDatasetConfig, self).__init__(LINK_MODE, name)

    @track(_get_logger, custom_dimensions={'app_name': 'LinkTabularOutputDatasetConfig'}, activity_type=_PUBLIC_API)
    def as_input(self, name=None):
        """Specify how to consume the output as an input in subsequent pipeline steps.

        :param name: The name of the input specific to the run.
        :type name: str
        :return: A :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` instance describing
            how to deliver the input data.
        :rtype: azureml.data.dataset_consumption_config.DatasetConsumptionConfig
        """
        from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

        name = name or self.__class__._generate_random_name('input')
        return DatasetConsumptionConfig(name, self, DIRECT_MODE)


class UploadOptions:
    """Options that are specific to output which will be uploaded."""

    def __init__(self, overwrite=False, source_globs=None):
        """Initialize a UploadOptions.

        :param overwrite: Whether to overwrite files that already exists in the destination.
        :type overwrite: bool
        :param source_globs: Glob patterns used to filter files that will be uploaded.
        :type source_globs: builtin.list[str]
        """
        if source_globs is not None and not isinstance(source_globs, list):
            source_globs = [source_globs]
        self.overwrite = overwrite
        self.source_globs = source_globs


class MountOptions:
    """Options that are specific to output which will be mounted."""

    def __init__(self, disable_metadata_cache=False):
        """Initialize a MountOptions.

        :param disable_metadata_cache: Whether to cache metadata in local node,
            if disabled a node will not be able to see files generated from other nodes during job running.
        """
        self.disable_metadata_cache = disable_metadata_cache


class RegistrationConfiguration:
    """Configuration that specifies how to register the output as a Dataset."""

    def __init__(self, name, description, tags):
        """Initialize a RegistrationConfiguration.

        :param name: The Dataset name to register the output under.
        :type name: str
        :param description: The description for the Dataset.
        :type description: str
        :param tags: A list of tags to be assigned to the Dataset.
        :type tags: dict[str, str]
        """
        self.name = name
        self.description = description
        self.tags = tags


class _Origin:
    """A special class that will not be copied/deepcopied.

    The purpose of this class is to have the ability to track things that will not be copied/deepcopied when
    copy and deepcopy is called. OutputDatasetConfig follows dataflow's pattern of creating a new copy of itself
    whenever any methods are called, this is desirable in most cases but for building pipeline, pipeline by default
    captures closures, this means you don't need to explicitly specify the order of steps and the pipeline SDK
    is able to figure this out on its own. To do so, when an output is passed to a step, it sets it producer_step in
    the output to that step, and later when it constructs the graph, the input instance of this output needs to figure
    out which output the input came from, it does this by querying its producer_step property. This won't work
    if we deep copy everything, which is why we have this class which overrides the behavior of copy and deepcopy
    so these things are not copied and shared between all the copied instances.
    """

    def __init__(self):
        self.producer_step = None

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


def _create_tabular_dataset_with_dataflow(output_file_dataset, dataflow):
    copy = deepcopy(output_file_dataset)
    copy._additional_transformations = dataflow
    return OutputTabularDatasetConfig(output_file_dataset_config=copy)
