# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for referencing single or multiple files in datastores or public URLs.

For more information, see the article [Add & register
datasets](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets).
To get started working with a file dataset, see https://aka.ms/filedataset-samplenotebook.
"""

import os
import re
import sys
import tempfile
import uuid
import urllib.parse
import hashlib
import json

from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml._common.exceptions import AzureMLException
from azureml._tracing import get_tracer
from azureml.data._dataprep_helper import (dataprep, dataprep_fuse,
                                           get_dataflow_for_execution, update_metadata)
from azureml.data._loggerfactory import _LoggerFactory, track, trace_warn, trace, trace_error
from azureml.data.abstract_dataset import AbstractDataset, _get_path_from_step
from azureml.data.constants import _PUBLIC_API
from azureml.data.dataset_error_handling import (
    _construct_message_and_check_exception_type, _dataprep_error_handler,
    _try_execute, _download_error_handler)
from azureml.exceptions import UserErrorException

_logger = None
_tracer = None  # type/: Optional[AmlTracer]


def _get_logger():
    global _logger
    if _logger is None:
        _logger = _LoggerFactory.get_logger(__name__)
    return _logger


def _get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = get_tracer(__name__)
    return _tracer


class FileDataset(AbstractDataset):
    """Represents a collection of file references in datastores or public URLs to use in Azure Machine Learning.

    A FileDataset defines a series of lazily-evaluated, immutable operations to load data from the
    data source into file streams. Data is not loaded from the source until FileDataset is asked to deliver data.

    A FileDataset is created using the :func:`azureml.data.dataset_factory.FileDatasetFactory.from_files` method
    of the FileDatasetFactory class.

    For more information, see the article `Add & register
    datasets <https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets>`_.
    To get started working with a file dataset, see https://aka.ms/filedataset-samplenotebook.

    .. remarks::

        FileDataset can be used as input of an experiment run. It can also be registered to workspace
        with a specified name and be retrieved by that name later.

        FileDataset can be subsetted by invoking different subsetting methods available on this class.
        The result of subsetting is always a new FileDataset.

        The actual data loading happens when FileDataset is asked to deliver the data into another
        storage mechanism (e.g. files downloaded or mounted to local path).
    """

    def __init__(self):
        """Initialize the FileDataset object.

        This constructor is not supposed to be invoked directly. Dataset is intended to be created using
        :class:`azureml.data.dataset_factory.FileDatasetFactory` class.
        """
        super().__init__()

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def to_path(self):
        """Get a list of file paths for each file stream defined by the dataset.

        .. remarks::
            The file paths are relative paths for local files when the file streams are downloaded or mounted.

            A common prefix will be removed from the file paths based on how data source was
            specified to create the dataset. For example:

            .. code-block:: python

                datastore = Datastore.get(workspace, 'workspaceblobstore')
                dataset = Dataset.File.from_files((datastore, 'animals/dog/year-*/*.jpg'))
                print(dataset.to_path())

                # ['year-2018/1.jpg'
                #  'year-2018/2.jpg'
                #  'year-2019/1.jpg']

                dataset = Dataset.File.from_files('https://dprepdata.blob.core.windows.net/demo/green-small/*.csv')

                print(dataset.to_path())
                # ['/green_tripdata_2013-08.csv']

        :return: Returns an array of file paths.
        :rtype: builtin.list(str)
        """
        return self._to_path(activity='to_path')

    def _to_path(self, activity):
        dataflow, portable_path = _add_portable_path_column(self._dataflow)
        dataflow = get_dataflow_for_execution(
            dataflow, activity, 'FileDataset')
        records = dataflow._to_pyrecords()
        return [r[portable_path] for r in records]

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def download(self, target_path=None, overwrite=False, ignore_not_found=False):
        """Download file streams defined by the dataset as local files.

        .. remarks::

            If target_path starts with a /, then it will be treated as an absolute path. If it doesn't start
            with a /, then it will be treated as a relative path relative to the current working directory.

        :param target_path: The local directory to download the files to. If None, the data will be downloaded
            into a temporary directory.
        :type target_path: str
        :param overwrite: Indicates whether to overwrite existing files. The default is False. Existing files will
            be overwritten if overwrite is set to True; otherwise an exception will be raised.
        :type overwrite: bool
        :param ignore_not_found: Indicates whether to fail download if some files pointed to by dataset are not found.
            The default is False. Download will fail if any file download fails for any reason if ignore_not_found is
            set to False; otherwise a waring will be logged for not found errors and dowload will succeed as long as
            no other error types are encountered.
        :type ignore_not_found: bool
        :return: Returns an array of file paths for each file downloaded.
        :rtype: builtin.list(str)
        """
        with _get_tracer().start_as_current_span('download', user_facing_name='Dataset.download') as span:
            target_path, is_empty = _ensure_path(target_path)
            if self.id:
                span.set_user_facing_attribute('dataset_id', self.id)
            span.set_user_facing_attribute('target_path', target_path)

            download_list = None
            if not is_empty and not overwrite:
                # need to fail if destination is not empty and download would have collisions
                download_list = [
                    os.path.abspath(os.path.join(target_path, '.' + p))
                    for p in self._to_path(activity='download.to_path')
                ]

                for p in download_list:
                    # encode p to avoid UnicodeEncodeError from os.path.exists
                    if os.path.exists(_encode_if_needed(p)):
                        raise UserErrorException('File "{}" already exists. Set overwrite=True to overwrite it, \
                            or choose an empty target path.'.format(p))
            # at this point the folder was either empty or there were no collisions so we can proceed.
            base_path = dataprep().api.datasources.LocalFileOutput(target_path)

            dataflow, portable_path = _add_portable_path_column(self._dataflow)
            dataflow = dataflow.write_streams(
                streams_column='Path',
                base_path=base_path,
                file_names_column=portable_path)

            dataflow = get_dataflow_for_execution(
                dataflow, 'download', 'FileDataset')

            download_records = _try_execute(
                dataflow._to_pyrecords,
                'download',
                None if self.id is None else {'id': self.id, 'name': self.name, 'version': self.version})
            download_list = _get_and_validate_download_list(download_records,
                                                            download_list,
                                                            target_path,
                                                            ignore_not_found)
            return download_list

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def mount(self, mount_point=None, **kwargs):
        """Create a context manager for mounting file streams defined by the dataset as local files.

        .. remarks::

            A context manager will be returned to manage the lifecycle of the mount. To mount, you will need to
            enter the context manager and to unmount, exit from the context manager.

            Mount is only supported on Unix or Unix-like operating systems with the native package libfuse installed.
            If you are running inside a docker container, the docker container must be started with the `--privileged`
            flag or started with `--cap-add SYS_ADMIN --device /dev/fuse`.

           .. code-block:: python

                datastore = Datastore.get(workspace, 'workspaceblobstore')
                dataset = Dataset.File.from_files((datastore, 'animals/dog/year-*/*.jpg'))

                with dataset.mount() as mount_context:
                    # list top level mounted files and folders in the dataset
                    os.listdir(mount_context.mount_point)

                # You can also use the start and stop methods
                mount_context = dataset.mount()
                mount_context.start()  # this will mount the file streams
                mount_context.stop()  # this will unmount the file streams

           If target_path starts with a /, then it will be treated as an absolute path. If it doesn't start
           with a /, then it will be treated as a relative path relative to the current working directory.

        :param mount_point: The local directory to mount the files to. If None, the data will be mounted into a
            temporary directory, which you can find by calling the `MountContext.mount_point` instance method.
        :type mount_point: str
        :return: Returns a context manager for managing the lifecycle of the mount.
        :rtype: MountContext: the context manager. Upon entering the context manager, the dataflow will be
            mounted to the mount_point. Upon exit, it will remove the mount point and clean up the daemon process
            used to mount the dataflow.
        """
        force_clex = False
        allow_fallback_to_clex = True
        _CLEX_MOUNT = '_CLEX_MOUNT'
        if _CLEX_MOUNT in os.environ:
            if os.environ[_CLEX_MOUNT] == 'True':
                force_clex = True
            elif os.environ[_CLEX_MOUNT] == 'False':
                allow_fallback_to_clex = False
        try:
            dprep_fuse = dataprep_fuse()
        except OSError as e:
            raise UserErrorException('Mount is only supported on Unix or Unix-like operating systems with the '
                                     'native package libfuse installed. For more information, please refer to the '
                                     'remarks section of FileDataset.mount\'s documentation. Execution failed'
                                     'unexpectedly due to {}'.format(e.__class__.__name__))
        except Exception as e:
            raise AzureMLException(
                "Mount failed unexpectedly due to: {}".format(str(e)))

        mount_point, is_empty = _ensure_path(mount_point)
        if os.path.ismount(mount_point):
            raise UserErrorException('"{0}" is already mounted. Run `sudo umount "{0}"` to unmount it.'
                                     .format(mount_point))
        if not is_empty:
            raise UserErrorException('Dataset mount point must be empty, mounting dataset to non-empty folder \
                is not supported.')

        invocation_id = str(uuid.uuid4())

        was_optimized = False
        # dataflow was not loaded from json string yet and we are allowed to attempt RSLEX mount
        if(type(self._definition) == str) and not force_clex:
            try:
                # attempt to load dataflow and optimize simple case for file dataset to avoid going trough CLEX
                dataflow_json = json.loads(self._definition)
                if(len(dataflow_json['blocks']) == 1
                   and dataflow_json['blocks'][0]['type'] == 'Microsoft.DPrep.GetDatastoreFilesBlock'):
                    datastores = dataflow_json['blocks'][0]['arguments']
                    from azureml.dataprep.api.dataflow import Dataflow
                    from azureml.dataprep.api.step import Step
                    from azureml.dataprep.api.engineapi.typedefinitions import PropertyValues
                    # constuct non-functional dataflow without engineapi, this is only good to be passed in to volume
                    # mount as it would be deconstructed anyway
                    dataflow = Dataflow(None,
                                        [Step('Microsoft.DPrep.GetDatastoreFilesBlock',
                                              PropertyValues.from_pod(datastores,
                                                                      [{'name': 'datastores', 'type': 1}]))])
                    dataflow._meta = update_metadata(dataflow, 'mount', 'FileDataset', invocation_id=invocation_id)
                    trace(_logger, 'Optimized dataflow to avoid CLEX load')
                    was_optimized = True
            except Exception as e:
                trace(_logger, f'Attempt to optimize dataflow to avoid CLEX load failes with: {e}')
                pass
        if not was_optimized:
            dataflow = get_dataflow_for_execution(self._dataflow, 'mount', 'FileDataset',
                                                  invocation_id=invocation_id)
        mount_options = kwargs.get('mount_options', None)
        skip_validate = kwargs.get('skip_validate', False)
        client_id = kwargs.get('client_id', None)
        identity_endpoint_type = kwargs.get('identity_endpoint_type', None)
        enable_rslex_mount = kwargs.get('enable_rslex_mount', None)

        from azureml.dataprep.fuse.dprepfuse import rslex_direct_volume_mount, clex_mount, rslex_uri_volume_mount

        volume_mount_not_supported = False
        if force_clex:
            trace(
                _logger, f'env variable {_CLEX_MOUNT} is set to True hence skip rslex volumn mount')
        else:
            trace(_logger, 'rslex direct volumn mount start!')
            try:
                mount_context = rslex_direct_volume_mount(
                    dataflow=dataflow,
                    mount_point=mount_point,
                    options=mount_options,
                    client_id=client_id,
                    identity_endpoint_type=identity_endpoint_type,
                    enable_rslex_mount=enable_rslex_mount,
                    invocation_id=invocation_id)
                trace(_logger, 'rslex direct volumn mount success!')
                return mount_context
            except dprep_fuse.VolumeMountNotSupported as e:
                message = str(e)
                trace(_logger, 'Failed to run rslex based mount due to exception of type {} with message {}.'
                               'Will try uri mount.'.format(type(e).__name__, message))
                print('Not mounting as a volume: {}. \nWill try to uri mount.'.format(message))
                volume_mount_not_supported = True
                pass
            except dprep_fuse.VolumeMountNotEnabled:
                trace(_logger, "RslexDirectVolumeMount is not enabled")
                print('Volume mount is not enabled. \nFalling back to dataflow mount.')
                pass
            except dprep_fuse.VolumeMountFailed as e:
                trace_warn(_logger, 'Error during volume mount: {}'.format(e))
                if not skip_validate:
                    raise UserErrorException("Cannot mount Dataset(id='{}', name='{}', version={}). "
                                             "Error Message: {}".format(self.id, self.name, self.version, str(e)))
                else:
                    # when skip validate is passed we should mount even if there is no data, so falling back to clex
                    pass
            except BaseException as e:
                message = str(e)
                trace_warn(
                    _logger, 'Unexpected error during volume mount: {}'.format(message))

            # rslex mount has failed, so we have to undo optimization attempt and get fully functional dataflow again
            trace(_logger, 'Direct volume mount with RSLEX has failed, CLEX will be loaded.')
            dataflow = get_dataflow_for_execution(self._dataflow, 'mount', 'FileDataset',
                                                  invocation_id=invocation_id)
            if volume_mount_not_supported:
                trace(_logger, 'rslex volume uri mount for dataflow start!')
                try:
                    from azureml.dataprep.api._dataflow_script_resolver import resolve_dataflow

                    dataflow_yaml = resolve_dataflow(dataflow)
                    hash_object = hashlib.md5(dataflow_yaml.encode()).hexdigest()
                    dataflow_in_memory_uri = f'inmemory://dataflow/{hash_object}'
                    dataflow_in_memory_uri_encoded = urllib.parse.quote(
                        dataflow_in_memory_uri.encode('utf8'), safe='')
                    stream_column_encode = urllib.parse.quote(
                        'Path'.encode('utf8'), safe='')
                    dataflow_uri = f"rsdf://dataflowfs/{dataflow_in_memory_uri_encoded}/{stream_column_encode}/"

                    from azureml.dataprep.rslex import add_in_memory_stream
                    add_in_memory_stream(dataflow_in_memory_uri, dataflow_yaml)
                    mount_context = rslex_uri_volume_mount(
                        uri=dataflow_uri, mount_point=mount_point, options=mount_options)
                    trace(_logger, 'rslex volumn uri mount for dataflow success!')
                    return mount_context
                except BaseException as e:
                    message = str(e)
                    if any(errorName in message for errorName in ["InvalidURIScheme",
                                                                  "StreamError(NotFound)",
                                                                  "DataAccessError(NotFound)",
                                                                  "DataAccessError(PermissionDenied)"]):
                        trace_error(_logger, message)
                        raise UserErrorException(message)
                    else:
                        trace(_logger, 'Failed to run rslex uri based mount due to exception of type {} with'
                              'message {}. Falling back to dataflow mount.'.format(type(e).__name__, message))
                        print('Not mounting as a volume: {}. \nFalling back to dataflow mount.'.format(
                            message))
                        pass

        if allow_fallback_to_clex:
            (base_path, was_data_pulled) = _find_path_prefix(
                dataflow, skip_validate)
            if not skip_validate and not was_data_pulled:
                # only need to validate if we are allowed to pull data and haven't already pulled it for the prefix
                try:
                    is_invalid = dataflow.has_invalid_source(
                        return_validation_error=True)
                    if is_invalid is not False:  # This means that the source is invalid
                        raise UserErrorException("Cannot mount Dataset(id='{}', name='{}', version={}). "
                                                 "Source of the dataset is either not "
                                                 "accessible or does not contain any data. "
                                                 "Error Message: {}".format(self.id, self.name, self.version,
                                                                            is_invalid))
                except TypeError:
                    # This catch is for backwards compatibility. There are valid version combinations of dataprep
                    # and core where dataflow.has_invalid_source will not have the return_validation_error parameter,
                    # which the above call will throw a TypeError.
                    if dataflow.has_invalid_source():  # This means that the source is invalid
                        raise UserErrorException("Cannot mount dataset. Source of the dataset is either not "
                                                 "accessible or does not contain any data. ")
                except AttributeError:
                    # This catch is for backwards compatibility. There are valid version combinations of dataprep
                    # and core where Dataflow will not have the has_invalid_source method.
                    pass
                except UserErrorException:
                    raise
                except AzureMLException:
                    raise
                except Exception as e:
                    dataset_info = None if self.id is None else {
                        'id': self.id, 'name': self.name, 'version': self.version}
                    message, is_dprep_exception = _construct_message_and_check_exception_type(
                        e, dataset_info, "mount")
                    trace_error(_logger, message)
                    _dataprep_error_handler(e, message, is_dprep_exception)
            else:
                trace(_logger, 'skip_validate is set, will mount with CLex.')
            return clex_mount(
                dataflow=dataflow,
                files_column='Path',
                mount_point=mount_point,
                base_path=base_path,
                options=mount_options,
                foreground=False,
                invocation_id=invocation_id,
                client_id=client_id,
                identity_endpoint_type=identity_endpoint_type,
                enable_rslex_mount=enable_rslex_mount)
        else:
            trace_error(
                _logger, f"env variable {_CLEX_MOUNT} is set to False hence fallback to clex mount was disabled")
            raise UserErrorException(
                f"env variable {_CLEX_MOUNT} is set to False hence fallback to clex mount was disabled")

    def as_mount(self, path_on_compute=None):
        """Create a DatasetConsumptionConfig with the mode set to mount.

        In the submitted run, files in the datasets will be mounted to local path on the compute target.
        The mount point can be retrieved from argument values and the input_datasets field of the run context.
        We will automatically generate an input name. If you would like specify a custom input name, please call
        the as_named_input method.

        .. code-block:: python

            # Given a run submitted with dataset input like this:
            dataset_input = dataset.as_mount()
            experiment.submit(ScriptRunConfig(source_directory, arguments=[dataset_input]))


            # Following are sample codes running in context of the submitted run:

            # The mount point can be retrieved from argument values
            import sys
            mount_point = sys.argv[1]

            # The mount point can also be retrieved from input_datasets of the run context.
            from azureml.core import Run
            mount_point = Run.get_context().input_datasets['input_1']

        .. remarks::

            When the dataset is created from path of a single file, the mount point will be path of the single mounted
            file. Otherwise, the mount point will be path of the enclosing folder for all the mounted files.

            If path_on_compute starts with a /, then it will be treated as an absolute path. If it doesn't start
            with a /, then it will be treated as a relative path relative to the working directory. If you have
            specified an absolute path, please make sure that the job has permission to write to that directory.

        :param path_on_compute: The target path on the compute to make the data available at.
        :type path_on_compute: str
        """
        return (self
                .as_named_input(name=None)
                .as_mount(path_on_compute=path_on_compute))

    def as_download(self, path_on_compute=None):
        """Create a DatasetConsumptionConfig with the mode set to download.

        In the submitted run, files in the dataset will be downloaded to local path on the compute target.
        The download location can be retrieved from argument values and the input_datasets field of the run context.
        We will automatically generate an input name. If you would like specify a custom input name, please call
        the as_named_input method.

        .. code-block:: python

            # Given a run submitted with dataset input like this:
            dataset_input = dataset.as_download()
            experiment.submit(ScriptRunConfig(source_directory, arguments=[dataset_input]))


            # Following are sample codes running in context of the submitted run:

            # The download location can be retrieved from argument values
            import sys
            download_location = sys.argv[1]

            # The download location can also be retrieved from input_datasets of the run context.
            from azureml.core import Run
            download_location = Run.get_context().input_datasets['input_1']

        .. remarks::

            When the dataset is created from path of a single file, the download location will be path of the single
            downloaded file. Otherwise, the download location will be path of the enclosing folder for all the
            downloaded files.

            If path_on_compute starts with a /, then it will be treated as an absolute path. If it doesn't start
            with a /, then it will be treated as a relative path relative to the working directory. If you have
            specified an absolute path, please make sure that the job has permission to write to that directory.

        :param path_on_compute: The target path on the compute to make the data available at.
        :type path_on_compute: str
        """
        return (self
                .as_named_input(name=None)
                .as_download(path_on_compute=path_on_compute))

    def as_hdfs(self):
        """Set the mode to hdfs.

        In the submitted synapse run, files in the datasets will be converted to local path on the compute target.
        The hdfs path can be retrieved from argument values and the os environment variables.

        .. code-block:: python

            # Given a run submitted with dataset input like this:
            dataset_input = dataset.as_hdfs()
            experiment.submit(ScriptRunConfig(source_directory, arguments=[dataset_input]))


            # Following are sample codes running in context of the submitted run:

            # The hdfs path can be retrieved from argument values
            import sys
            hdfs_path = sys.argv[1]

            # The hdfs path can also be retrieved from input_datasets of the run context.
            import os
            hdfs_path = os.environ['input_<hash>']

        .. remarks::

            When the dataset is created from path of a single file, the hdfs path will be path of the single
            file. Otherwise, the hdfs path will be path of the enclosing folder for all the mounted files.

        """
        return (self
                .as_named_input(name=None)
                .as_hdfs())

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def skip(self, count):
        """Skip file streams from the top of the dataset by the specified count.

        :param count: The number of file streams to skip.
        :type count: int
        :return: Returns a new FileDataset object representing a dataset with file streams skipped.
        :rtype: azureml.data.FileDataset
        """
        return FileDataset._create(self._dataflow.skip(count), self._properties, telemetry_info=self._telemetry_info)

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def take(self, count):
        """Take a sample of file streams from top of the dataset by the specified count.

        :param count: The number of file streams to take.
        :type count: int
        :return: Returns a new FileDataset object representing the sampled dataset.
        :rtype: azureml.data.FileDataset
        """
        return FileDataset._create(self._dataflow.take(count), self._properties, telemetry_info=self._telemetry_info)

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def take_sample(self, probability, seed=None):
        """Take a random sample of file streams in the dataset approximately by the probability specified.

        :param probability: The probability of a file stream being included in the sample.
        :type probability: float
        :param seed: An optional seed to use for the random generator.
        :type seed: int
        :return: Returns a new FileDataset object representing the sampled dataset.
        :rtype: azureml.data.FileDataset
        """
        return FileDataset._create(
            self._dataflow.take_sample(probability, seed), self._properties, telemetry_info=self._telemetry_info)

    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def random_split(self, percentage, seed=None):
        """Split file streams in the dataset into two parts randomly and approximately by the percentage specified.

        The first dataset returned contains approximately ``percentage`` of the total number of file references
        and the second dataset contains the remaining file references.

        :param percentage: The approximate percentage to split the dataset by. This must be a number between 0.0
            and 1.0.
        :type percentage: float
        :param seed: An optional seed to use for the random generator.
        :type seed: int
        :return: Returns a tuple of new FileDataset objects representing the two datasets after the split.
        :rtype: (azureml.data.FileDataset, azureml.data.FileDataset)
        """
        dataflow1, dataflow2 = self._dataflow.random_split(percentage, seed)
        return (
            FileDataset._create(dataflow1, self._properties,
                                telemetry_info=self._telemetry_info),
            FileDataset._create(dataflow2, self._properties,
                                telemetry_info=self._telemetry_info)
        )

    @experimental
    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def file_metadata(self, col):
        """Get file metadata expression by specifying the metadata column name.

        Supported file metadata columns are Size, LastModifiedTime, CreationTime, Extension and CanSeek

        :param col: Name of column
        :type col: str
        :return: Returns an expression that retrieves the value in the specified column.
        :rtype: azureml.dataprep.api.expression.RecordFieldExpression
        """
        from azureml.dataprep.api.functions import get_stream_properties
        return get_stream_properties(self._dataflow['Path'])[col]

    @experimental
    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def filter(self, expression):
        """
        Filter the data, leaving only the records that match the specified expression.

        .. remarks::

            Expressions are started by indexing the Dataset with the name of a column. They support a variety of
                functions and operators and can be combined using logical operators. The resulting expression will be
                lazily evaluated for each record when a data pull occurs and not where it is defined.

            .. code-block:: python

                (dataset.file_metadata('Size') > 10000) & (dataset.file_metadata('CanSeek') == True)
                dataset.file_metadata('Extension').starts_with('j')

        :param expression: The expression to evaluate.
        :type expression: azureml.dataprep.api.expression.Expression
        :return: The modified dataset (unregistered).
        :rtype: azureml.data.FileDataset
        """
        dataflow = self._dataflow
        dataflow = dataflow.filter(expression)
        return FileDataset._create(dataflow, self._properties, telemetry_info=self._telemetry_info)

    @experimental
    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def as_cache(self, datacache_store):
        """
        Create a DatacacheConsumptionConfig mapped to a datacache_store and a dataset.

        :param datacache_store: The datacachestore to be used to hydrate.
        :type datacache_store: azureml.data.datacache.DatacacheStore
        :return: The configuration object describing how the datacache should be materialized in the run.
        :rtype: azureml.data.datacache_consumption_config.DatacacheConsumptionConfig
        """
        from azureml.data.datacache import _Datacache
        from azureml.data.datacache_consumption_config import DatacacheConsumptionConfig

        dc = _Datacache.create(datacache_store.workspace,
                               datacache_store, self)
        return DatacacheConsumptionConfig(datacache_store=dc.datacache_store,
                                          dataset=dc.dataset,
                                          _datacache_id=dc._id)

    @experimental
    @track(_get_logger, custom_dimensions={'app_name': 'FileDataset'}, activity_type=_PUBLIC_API)
    def hydrate(self, datacache_store, replica_count=None):
        """
        Hydrate the dataset into the requested replicas specified in datacache_store.

        :param datacache_store: The datacachestore to be used to hydrate.
        :type datacache_store: azureml.data.datacache.DatacacheStore
        :param replica_count: Number of replicas to hydrate.
        :type replica_count: Int, optional
        :return: The configuration object describing how the datacache should be materialized in the run.
        :rtype: azureml.data.datacache.DatacacheHydrationTracker
        """
        from azureml.data.datacache import _Datacache
        dc = _Datacache.create(datacache_store.workspace,
                               datacache_store, self)
        return dc.hydrate(replica_count)


def _add_portable_path_column(dataflow):
    (prefix_path, _) = _find_path_prefix(dataflow)
    portable_path = 'Portable Path'
    get_portable_path = dataprep().api.functions.get_portable_path
    col = dataprep().api.expressions.col
    return dataflow.add_column(get_portable_path(col('Path'), prefix_path), portable_path, 'Path'), portable_path


def _find_path_prefix(dataflow, avoid_data_pull=False):
    steps = dataflow._get_steps()
    step_types = [s.step_type for s in steps]
    special_block_types = {'Microsoft.DPrep.ToCsvStreamsBlock',
                           'Microsoft.DPrep.ToParquetStreamsBlock',
                           'Microsoft.DPrep.ToDataFrameDirectoryBlock'}
    if len(special_block_types.intersection(step_types)) > 0:
        return (None, False)
    step_type = steps[0].step_type
    step_arguments = steps[0].arguments
    if hasattr(step_arguments, 'to_pod'):
        step_arguments = step_arguments.to_pod()
    path = _get_path_from_step(step_type, step_arguments)
    return (None, False) if path is None else _get_prefix(path, dataflow, avoid_data_pull)


def _get_prefix(path, dataflow, avoid_data_pull=False):
    """Determine if there exists a common prefix for all files which may exist under the given path/dataflow.

    :param path: Path extracted from dataflow
    :param dataflow: Dataflow to get prefix for.
    :return: Path which is common prefix of all files under path/dataflow, or None if a common prefix was not found.
    """
    from azureml.dataprep.api.errorhandlers import ExecutionError
    from azureml.dataprep.api.functions import get_portable_path
    from azureml.dataprep.native import DataPrepError

    if '*' in path:
        return ('/'.join(re.split(r'/|\\', path.split('*')[0])[:-1]), False)

    supported_pattern = re.compile(
        r'^(https?|adl|wasbs?|abfss?)://', re.IGNORECASE)
    if supported_pattern.match(path):
        return (path[:path.rindex('/')], False)

    if avoid_data_pull:
        return (None, False)
    dataflow = dataflow.add_column(get_portable_path(
        dataflow['Path']), 'PortablePath', 'Path')
    paths = []
    try:
        paths = [r['PortablePath'] for r in dataflow.take(1)._to_pyrecords()]
    except ExecutionError as e:
        if 'InvalidPath' in e.error_code or 'NotFound' in e.error_code:
            return (None, False)
        raise e
    if len(paths) == 0:
        return (None, False)
    if isinstance(paths[0], DataPrepError):
        raise UserErrorException('DataPrepError: {0}, '
                                 'please make sure the URI/Path is accessible'
                                 ' and the AML datastore is registered/accessible.'.format(paths[0]))
    if len(paths) == 1 and paths[0].endswith(path):
        return (path.replace('\\', '/')[:path.rindex('/')], True)
    return (path, True)


def _ensure_path(path):
    if not path or path.isspace():
        return (tempfile.mkdtemp(), True)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
            return (os.path.abspath(path), True)
        except FileExistsError:
            # There is a chance that the directory may be created after we check for existence and
            # before we create it. In this case, we can no-op as though the directory already existed.
            pass

    is_empty = True
    for _, dirnames, files in os.walk(path):
        if files or dirnames:
            is_empty = False
            break
    return (os.path.abspath(path), is_empty)


def _encode_if_needed(path):
    sys_encoding = sys.getfilesystemencoding() or sys.getdefaultencoding()
    try:
        path.encode(sys_encoding)
        return path  # no need to encode
    except (UnicodeError, LookupError):
        # Encode the path string when it contains characters which cannot be encoded by sys encoding.
        # Otherwise, usage of the path string (e.g. `os.path.exists(p)`) can encounter UnicodeEncodeError.
        return path.encode('utf8')


def _get_and_validate_download_list(download_records, download_list, target_path, ignore_not_found):
    if len(download_records) == 0:
        return []
    from azureml.dataprep.native import DataPrepError, StreamInfo
    # handle CLEX
    if 'Portable Path' in download_records[0]:
        if download_list:
            # CLEX returns the source list, so no new information here, just return back dowload list we got
            return download_list
        else:
            # we don't have dowload list so need to generate it based on portable path and target path.
            # capture DataPrepError
            error_list = []
            for record in download_records:
                value = record['Portable Path']
                if isinstance(value, DataPrepError):
                    resource_identifier = value.originalValue
                    # this is for backward compatibility as error used to have StreamInfo as original value previously
                    if isinstance(value.originalValue, StreamInfo):
                        resource_identifier = value.originalValue.resource_identifier

                    error_list.append((resource_identifier, value.errorCode))

            if error_list:
                message = 'Some files have failed to download:' + '\n'.join(
                    [str((file_name, error_code)) for (file_name, error_code) in error_list])
                for (_, error) in error_list:
                    trace_error(
                        _logger, "System error happens during downloading: {}".format(error))
                raise AzureMLException(message)

            return [os.path.abspath(os.path.join(target_path, '.' + p['Portable Path']))
                    for p in download_records]
    elif 'DestinationFile' in download_records[0]:
        # this means RsLEX download, so we actually get more info here, like errors and actual download paths.
        actual_download_list = []
        error_list = []
        for record in download_records:
            value = record['DestinationFile']
            if isinstance(value, StreamInfo):
                actual_download_list.append(value.resource_identifier)
            elif isinstance(value, DataPrepError):
                resource_identifier = value.originalValue
                # this is for backward compatibility as error used to have StreamInfo as original value previously
                if isinstance(value.originalValue, StreamInfo):
                    resource_identifier = value.originalValue.resource_identifier
                if ignore_not_found and value.errorCode == "Microsoft.DPrep.ErrorValues.SourceFileNotFound":
                    _log_and_print_warning("'{}' hasn't been downloaded as it was not present at the source. \
                        Download is proceeding.".format(resource_identifier))
                else:
                    error_list.append((resource_identifier, value.errorCode))
            else:
                raise AzureMLException(
                    'Unexpected error during file download.')

        if error_list:
            # this will throw UserErrorException or AzureMLException based on set of errors encountered
            _download_error_handler(error_list, _get_logger())
        return actual_download_list


def _log_and_print_warning(message):
    from datetime import datetime
    now = datetime.utcnow().isoformat(timespec='milliseconds')
    trace_warn(_logger, message)
    print(f"[{now}] {message}")
