# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
from functools import wraps
from contextlib import contextmanager
from azureml.data.constants import _DATASET_PROP_DEPRECATED_BLOCKS
from azureml.data._loggerfactory import _LoggerFactory


_warned = set()
_warning_silenced_for = None
_warning_logger = None
_telemetry_logger = None


def deprecated(target, replacement=None):
    def monitor(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _warning_silenced_for
            if _warning_silenced_for is None:
                _warn_deprecation(target, replacement)  # only raise warning for top-level invocation
                _warning_silenced_for = target
            result = func(*args, **kwargs)
            if _warning_silenced_for == target:
                _warning_silenced_for = None
            return result
        return wrapper
    return monitor


@contextmanager
def silent_deprecation_warning():
    global _warning_silenced_for
    _warning_silenced_for = '__WARNING_DISABLED__'
    try:
        yield
    finally:
        _warning_silenced_for = None


def warn_deprecated_blocks(dataset):
    try:
        if not dataset._properties:
            return
        deprecated_blocks = dataset._properties.get(_DATASET_PROP_DEPRECATED_BLOCKS)
        if not deprecated_blocks:
            return

        telemetry_dimensions = {
            'dataset_id': dataset.id,
            'deprecated_blocks': str(deprecated_blocks)
        }
        if dataset._registration and dataset._registration.workspace:
            workspace = dataset._registration.workspace
            telemetry_dimensions.update({
                'subscription_id': workspace.subscription_id,
                'resource_group_name': workspace.resource_group,
                'workspace_name': workspace.name,
                'location': workspace.location
            })
        _LoggerFactory.track_activity(_get_telemetry_logger(), 'warn_deprecated_blocks', telemetry_dimensions)

        dataset_info = 'id: {}'.format(dataset.id)
        if dataset.name:
            dataset_info += ', name: {}'.format(dataset.name)
        msg = (
            'This dataset ({}) was created using unsupported API and has deprecated definition. '
            'Some functionalities of the dataset may not be supported in future versions. '
            'To avoid using unsupported functionalities, please refer to [Create Azure Machine Learning '
            'datasets](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets) '
            'and the latest API reference to create the dataset again.\nDeprecation details:\n  {}'
        ).format(dataset_info, deprecated_blocks)
        _get_warning_logger().warning(msg)
    except Exception as e:
        _get_telemetry_logger().error('Failed to warn deprecated blocks due to: {}'.format(repr(e)))


def _warn_deprecation(target, replacement=None):
    global _warned
    if target in _warned:
        return  # only warn per target per session
    msg = '{} is deprecated after version 1.0.69.'.format(target)
    if replacement:
        msg += ' Please use {}.'.format(replacement)
    msg += ' See Dataset API change notice at https://aka.ms/dataset-deprecation.'
    _get_warning_logger().warning(msg)
    _warned.add(target)


def _get_warning_logger():
    # this logger is for printing warning message
    global _warning_logger
    if _warning_logger is None:
        _warning_logger = logging.getLogger(__name__)
    return _warning_logger


def _get_telemetry_logger():
    # this logger is for telemetry logging
    global _telemetry_logger
    if _telemetry_logger is None:
        _telemetry_logger = _LoggerFactory.get_logger(__name__)
    return _telemetry_logger
