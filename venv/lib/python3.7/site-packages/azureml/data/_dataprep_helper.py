# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains helper methods for dataprep."""

import sys

from azureml._base_sdk_common.utils import common_path

MIN_DATAPREP_VERSION = '1.1.29'
_version_checked = False


def check_min_version():
    global _version_checked
    if _version_checked:
        return
    _version_checked = True
    from pkg_resources import parse_version, get_distribution
    import logging
    installed_version = get_distribution('azureml-dataprep').version
    if parse_version(installed_version) < parse_version(MIN_DATAPREP_VERSION):
        logging.getLogger().warning(
            _dataprep_incompatible_version_error.format(MIN_DATAPREP_VERSION, installed_version))


def is_dataprep_installed():
    try:
        from azureml.dataprep import api
        return api is not None
    except Exception:
        return False


def dataprep():
    if not is_dataprep_installed():
        raise ImportError(get_dataprep_missing_message())
    import azureml.dataprep as _dprep
    check_min_version()
    return _dprep


def dataprep_fuse():
    try:
        import azureml.dataprep.fuse.dprepfuse as _dprep_fuse
        check_min_version()
        return _dprep_fuse
    except ImportError:
        raise ImportError(get_dataprep_missing_message(extra='[fuse]'))


def ensure_dataflow(dataflow):
    if not isinstance(dataflow, dataprep().Dataflow):
        raise RuntimeError('dataflow must be instance of azureml.dataprep.Dataflow')


def update_metadata(dataflow, action, source, **kwargs):
    from copy import deepcopy
    meta = deepcopy(dataflow._meta)

    if 'activityApp' not in meta:
        meta['activityApp'] = source

    if 'activity' not in meta:
        meta['activity'] = action

    try:
        import os
        run_id = os.environ.get("AZUREML_RUN_ID", None)
        if run_id is not None:
            meta['runId'] = run_id  # keep this here so not to break existing reporting
            meta['run_id'] = run_id
    except Exception:
        pass

    if len(kwargs) > 0:
        kwargs.update(meta)
        meta = kwargs
    return meta


def get_dataflow_for_execution(dataflow, action, source, **kwargs):
    meta = update_metadata(dataflow, action, source, **kwargs)
    new_dataflow = dataprep().Dataflow(dataflow._engine_api, dataflow._steps, meta)
    new_dataflow._rs_dataflow_yaml = dataflow._rs_dataflow_yaml if hasattr(dataflow, '_rs_dataflow_yaml') else None
    return new_dataflow


def get_dataflow_with_meta_flags(dataflow, **kwargs):
    from copy import deepcopy
    if len(kwargs) > 0:
        meta = deepcopy(dataflow._meta)
        kwargs.update(meta)
        meta = kwargs
        return dataprep().Dataflow(dataflow._engine_api, dataflow._steps, meta)
    return dataflow


def find_first_different_step(left, right):
    """Compares the two dataflow and return the first index where the step differs.

    :param left: The dataflow to compare.
    :type left: azureml.dataprep.Dataflow
    :param right: The dataflow to compare.
    :type right: azureml.dataprep.Dataflow
    :return: int
    """
    left_steps = left._get_steps()
    right_steps = right._get_steps()
    short_length = min(len(left_steps), len(right_steps))
    for i in range(short_length):
        if not steps_equal(left_steps[i], right_steps[i]):
            return i
    return short_length


def steps_equal(left, right):
    import json

    to_dprep_pod = dataprep().api.engineapi.typedefinitions.to_dprep_pod
    CustomEncoder = dataprep().api.engineapi.typedefinitions.CustomEncoder

    if left.step_type != right.step_type:
        return False
    if json.dumps(to_dprep_pod(left.arguments), cls=CustomEncoder, sort_keys=True) \
            != json.dumps(to_dprep_pod(right.arguments), cls=CustomEncoder, sort_keys=True):
        return False
    if json.dumps(to_dprep_pod(left.local_data), cls=CustomEncoder, sort_keys=True) \
            != json.dumps(to_dprep_pod(right.local_data), cls=CustomEncoder, sort_keys=True):
        return False
    return True


def get_dataprep_missing_message(issue=None, extra=None, how_to_fix=None):
    dataprep_available = sys.maxsize > 2**32  # no azureml-dataprep available on 32 bit
    extra = extra or ''
    if how_to_fix:
        suggested_fix = ' This can {}be resolved by {}.'.format('also ' if dataprep_available else '', how_to_fix)
    else:
        suggested_fix = ''

    message = (issue + ' due to missing') if issue else 'Missing'
    message += ' required package "azureml-dataset-runtime{}", which '.format(extra)

    if not dataprep_available:
        message += 'is unavailable for 32bit Python.'
    else:
        message += 'can be installed by running: {}'.format(_get_install_cmd(extra))

    return message + suggested_fix


def is_tabular(transformations):
    tabular_transformations = {
        'Microsoft.DPrep.ReadParquetFileBlock',
        'Microsoft.DPrep.ParseDelimitedBlock',
        'Microsoft.DPrep.ParseJsonLinesBlock'
    }

    file_transformations = {
        'Microsoft.DPrep.ToCsvStreamsBlock',
        'Microsoft.DPrep.ToParquetStreamsBlock',
        'Microsoft.DPrep.ToDataFrameDirectoryBlock'
    }

    def r_index(collection, predicate):
        for i in range(len(collection) - 1, -1, -1):
            if predicate(collection[i]):
                return i
        return -1

    steps = transformations._get_steps()
    tabular_index = r_index(steps, lambda s: s.step_type in tabular_transformations)
    file_index = r_index(steps, lambda s: s.step_type in file_transformations)

    return tabular_index > file_index


def _get_engine_api():
    from azureml.dataprep.api.engineapi.api import get_engine_api

    if not hasattr(_get_engine_api, "engine_api"):
        _get_engine_api.engine_api = get_engine_api()

    return _get_engine_api.engine_api


def upload_dir(src_dir, remote_target_path, datastore, overwrite=False, show_progress=True, continue_on_failure=True):
    from azureml.data._upload_helper import _start_upload_task, _get_upload_from_dir
    from azureml.data.data_reference import DataReference
    from azureml.data.hdfs_datastore import HDFSDatastore

    # TASK 1714882: override _file_exists() and use specially crafted function for HDFS that returns False.
    # This bypasses file_exists check in CLex. Instead, RSLex will perform real file_exists checks for HDFS.
    file_exists_fn = _hdfs_file_exists if isinstance(datastore, HDFSDatastore) else _file_exists
    remote_target_path = remote_target_path or ""
    if not overwrite:
        file_exists_fn(dstore=datastore, path=remote_target_path)
    _start_upload_task(
        _get_upload_from_dir(src_dir, remote_target_path),
        overwrite,
        lambda target_file_path: file_exists_fn(dstore=datastore, path=target_file_path),
        show_progress,
        lambda target, source: lambda: _upload_file(base_path=src_dir, local_file_path=source,
                                                    remote_target_path=remote_target_path, datastore=datastore,
                                                    overwrite=overwrite),
        continue_on_failure
    )
    return DataReference(datastore=datastore, path_on_datastore=remote_target_path)


def upload_files(files,
                 datastore,
                 relative_root=None,
                 target_path=None,
                 overwrite=False,
                 show_progress=True,
                 continue_on_failure=True):
    from azureml.data._upload_helper import _start_upload_task, _get_upload_from_files
    from azureml.data.data_reference import DataReference
    from azureml.data.hdfs_datastore import HDFSDatastore

    # TASK 1714882: override _file_exists() and use specially crafted function for HDFS that returns False.
    # This bypasses file_exists check in CLex. Instead, RSLex will perform real file_exists checks for HDFS.
    file_exists_fn = _hdfs_file_exists if isinstance(datastore, HDFSDatastore) else _file_exists
    target_path = target_path or ""
    if not overwrite:
        file_exists_fn(dstore=datastore, path=target_path)
    relative_root = relative_root or common_path(files)
    _start_upload_task(
        _get_upload_from_files(files, target_path, relative_root, True),
        overwrite,
        lambda target_file_path: file_exists_fn(dstore=datastore, path=target_file_path),
        show_progress,
        lambda target, source: lambda: _upload_file(base_path=relative_root, local_file_path=source,
                                                    remote_target_path=target_path, datastore=datastore,
                                                    overwrite=overwrite),
        continue_on_failure
    )
    return DataReference(datastore=datastore, path_on_datastore=target_path)


def _upload_file(base_path, local_file_path, remote_target_path, datastore, overwrite=False):
    from azureml.dataprep.api.engineapi.typedefinitions import UploadFileMessageArguments
    from azureml.dataprep.api._datastore_helper import _to_stream_info_value

    _get_engine_api().upload_file(
        UploadFileMessageArguments(base_path=base_path,
                                   local_path=local_file_path,
                                   destination=_to_stream_info_value(datastore, remote_target_path),
                                   overwrite=overwrite))


def _file_exists(path, dstore):
    from azureml.dataprep.api.engineapi.typedefinitions import FileExistsMessageArguments
    from azureml.dataprep.api._datastore_helper import _to_stream_info_value

    return _get_engine_api().file_exists(
        FileExistsMessageArguments(remote_file_path=_to_stream_info_value(dstore, path)))


# TASK 1714882: Special handling of file exists functionality for HDFS
def _hdfs_file_exists(dstore, path):
    return False


def _get_install_cmd(extra):
    return '"{}" -m pip install azureml-dataset-runtime{} --upgrade'.format(sys.executable, extra or '')


_dataprep_incompatible_version_error = (
    'Warning: The minimum required version of "azureml-dataprep" is {}, but {} is installed.'
    + '\nSome functionality may not work correctly. Please upgrade it by running:'
    + '\n' + _get_install_cmd('[fuse,pandas]')
)
