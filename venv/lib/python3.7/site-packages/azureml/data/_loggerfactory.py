# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A log factory module that provides methods to log telemetric data for the Dataset SDK."""
import logging
import logging.handlers
import uuid
import json
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
COMPONENT_NAME = 'azureml.dataset'
session_id = 'l_' + str(uuid.uuid4())
instrumentation_key = ''
default_custom_dimensions = {'app_name': 'dataset'}
ActivityLoggerAdapter = None


try:
    from azureml.telemetry import get_telemetry_log_handler, INSTRUMENTATION_KEY, get_diagnostics_collection_info
    from azureml.telemetry.activity import log_activity as _log_activity, ActivityType, ActivityLoggerAdapter
    from azureml.telemetry.logging_handler import AppInsightsLoggingHandler
    from azureml._base_sdk_common import _ClientSessionId
    session_id = _ClientSessionId

    telemetry_enabled, verbosity = get_diagnostics_collection_info(component_name=COMPONENT_NAME)
    instrumentation_key = INSTRUMENTATION_KEY if telemetry_enabled else ''
    DEFAULT_ACTIVITY_TYPE = ActivityType.INTERNALCALL
except Exception:
    telemetry_enabled = False
    DEFAULT_ACTIVITY_TYPE = 'InternalCall'


class _LoggerFactory:
    _core_version = None
    _dataprep_version = None
    _spark_version = None

    @staticmethod
    def get_logger(name, verbosity=logging.DEBUG):
        logger = logging.getLogger(__name__).getChild(name)
        logger.propagate = False
        logger.setLevel(verbosity)
        if telemetry_enabled:
            if not _LoggerFactory._found_handler(logger, AppInsightsLoggingHandler):
                logger.addHandler(get_telemetry_log_handler(component_name=COMPONENT_NAME))

        return logger

    @staticmethod
    def track_activity(logger, activity_name, activity_type=DEFAULT_ACTIVITY_TYPE, input_custom_dimensions=None):
        _LoggerFactory._get_version_info()

        if input_custom_dimensions is not None:
            custom_dimensions = default_custom_dimensions.copy()
            custom_dimensions.update(input_custom_dimensions)
        else:
            custom_dimensions = default_custom_dimensions
        custom_dimensions.update({
            'source': COMPONENT_NAME,
            'version': _LoggerFactory._core_version,
            'dataprepVersion': _LoggerFactory._dataprep_version,
            'sparkVersion': _LoggerFactory._spark_version if _LoggerFactory._spark_version is not None else ''
        })

        run_info = _LoggerFactory._try_get_run_info()
        if run_info is not None:
            custom_dimensions.update(run_info)

        if telemetry_enabled:
            return _log_activity(logger, activity_name, activity_type, custom_dimensions)
        else:
            return _log_local_only(logger, activity_name, activity_type, custom_dimensions)

    @staticmethod
    def _found_handler(logger, handler_type):
        for log_handler in logger.handlers:
            if isinstance(log_handler, handler_type):
                return True
        return False

    @staticmethod
    def _get_version_info():
        if _LoggerFactory._core_version is not None and _LoggerFactory._dataprep_version is not None:
            return

        core_ver = _get_package_version('azureml-core')
        if core_ver is None:
            # only fallback when the approach above fails, as azureml.core.VERSION has no patch version segment
            try:
                from azureml.core import VERSION as core_ver
            except Exception:
                core_ver = ''
        _LoggerFactory._core_version = core_ver

        dprep_ver = _get_package_version('azureml-dataprep')
        if dprep_ver is None:
            try:
                from azureml.dataprep import __version__ as dprep_ver
            except Exception:
                # data-prep may not be installed
                dprep_ver = ''
        _LoggerFactory._dataprep_version = dprep_ver
        _LoggerFactory._spark_version = _get_package_version('pyspark')

    @staticmethod
    def _try_get_run_info():
        try:
            import re
            import os
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT")
            location = re.compile("//(.*?)\\.").search(location).group(1)
        except Exception:
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT", "")
        return {
            "subscription": os.environ.get('AZUREML_ARM_SUBSCRIPTION', ""),
            "run_id": os.environ.get("AZUREML_RUN_ID", ""),
            "resource_group": os.environ.get("AZUREML_ARM_RESOURCEGROUP", ""),
            "workspace_name": os.environ.get("AZUREML_ARM_WORKSPACE_NAME", ""),
            "experiment_id": os.environ.get("AZUREML_EXPERIMENT_ID", ""),
            "location": location
        }


def track(get_logger, custom_dimensions=None, activity_type=DEFAULT_ACTIVITY_TYPE):
    def monitor(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            with _LoggerFactory.track_activity(logger, func.__name__, activity_type, custom_dimensions) as al:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if hasattr(al, 'activity_info') and hasattr(e, 'error_code'):
                        al.activity_info['error_code'] = e.error_code
                        al.activity_info['outer_error_code'] = getattr(e, 'outer_error_code', '')
                        al.activity_info['message'] = getattr(e, 'compliant_message', '')
                    raise
                finally:
                    try:
                        if hasattr(al, 'activity_info'):  # OCV 2062331
                            if len(args) > 0 and (args[0].__class__.__name__ == 'FileDataset'
                                                  or args[0].__class__.__name__ == 'TabularDataset'):
                                if hasattr(args[0], "_registration") and args[0]._registration and \
                                        hasattr(args[0]._registration, "workspace") and \
                                        args[0]._registration.workspace:
                                    al.activity_info["subscription"] = args[0]._registration.workspace._subscription_id
                    except Exception as e:
                        logger.error('Failed to extract subscription information, Exception={}; {}'
                                     .format(type(e).__name__, str(e)))

        return wrapper

    return monitor


def trace(logger, message, custom_dimensions=None):
    import azureml.core
    core_version = azureml.core.VERSION

    run_info = _LoggerFactory._try_get_run_info()
    payload = dict(core_sdk_version=core_version)

    if run_info is not None:
        payload.update(run_info)
    payload.update(custom_dimensions or {})

    if ActivityLoggerAdapter:
        activity_logger = ActivityLoggerAdapter(logger, payload)
        activity_logger.info(message)
    else:
        logger.info('Message: {}\nPayload: {}'.format(message, json.dumps(payload)))


def trace_warn(logger, message, custom_dimensions=None):
    import azureml.core
    core_version = azureml.core.VERSION

    run_info = _LoggerFactory._try_get_run_info()
    payload = dict(core_sdk_version=core_version)

    if run_info is not None:
        payload.update(run_info)
    payload.update(custom_dimensions or {})

    if ActivityLoggerAdapter:
        activity_logger = ActivityLoggerAdapter(logger, payload)
        activity_logger.warning(message)
    else:
        logger.warning('Message: {}\nPayload: {}'.format(message, json.dumps(payload)))


def trace_error(logger, message, custom_dimensions=None):
    import azureml.core
    core_version = azureml.core.VERSION

    run_info = _LoggerFactory._try_get_run_info()
    payload = dict(core_sdk_version=core_version)

    if run_info is not None:
        payload.update(run_info)
    payload.update(custom_dimensions or {})

    if ActivityLoggerAdapter:
        activity_logger = ActivityLoggerAdapter(logger, payload)
        activity_logger.error(message)
    else:
        logger.error('Message: {}\nPayload: {}'.format(message, json.dumps(payload)))


def trace_debug(logger, message, custom_dimensions=None):
    import azureml.core
    core_version = azureml.core.VERSION

    run_info = _LoggerFactory._try_get_run_info()
    payload = dict(core_sdk_version=core_version)

    if run_info is not None:
        payload.update(run_info)
    payload.update(custom_dimensions or {})

    if ActivityLoggerAdapter:
        activity_logger = ActivityLoggerAdapter(logger, payload)
        activity_logger.debug(message)
    else:
        logger.debug('Message: {}\nPayload: {}'.format(message, json.dumps(payload)))


def collect_datasets_usage(logger, message, datasets, workspace, compute, custom_dimensions=None):
    common_dimensions = {
        'compute': compute if isinstance(compute, str) else type(compute).__name__
    }
    try:
        if workspace:
            try:
                common_dimensions.update({
                    'subscription_id': workspace.subscription_id,
                    'resource_group_name': workspace.resource_group,
                    'workspace_name': workspace.name,
                    'location': workspace.location
                })
            except Exception:
                pass

        common_dimensions.update(custom_dimensions or {})

        for ds_info in [_get_dataset_info(ds) for ds in datasets]:
            ds_info.update(common_dimensions)
            trace(logger, message, ds_info)
    except Exception as e:
        trace(logger, 'collect_datasets_usage FAILED with: ' + str(e))


def _get_dataset_info(dataset):
    from azureml.data._dataset import _Dataset
    from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
    if isinstance(dataset, _Dataset):
        return {
            'input_type': type(dataset).__name__,
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'dataset_version': dataset.version
        }
    if isinstance(dataset, DatasetConsumptionConfig):
        return {
            'input_type': type(dataset.dataset).__name__,
            'dataset_id': dataset.dataset.id,
            'dataset_name': dataset.dataset.name,
            'dataset_version': 'dataset.dataset.version',
            'consumption_mode': dataset.mode
        }
    return {'input_type': 'Unknown:' + type(dataset).__name__}


@contextmanager
def _log_local_only(logger, activity_name, activity_type, custom_dimensions):
    activity_info = dict(activity_id=str(uuid.uuid4()), activity_name=activity_name, activity_type=activity_type)
    custom_dimensions = custom_dimensions or {}
    activity_info.update(custom_dimensions)

    start_time = datetime.utcnow()
    completion_status = 'Success'

    message = 'ActivityStarted, {}'.format(activity_name)
    logger.info(message)
    exception = None

    try:
        yield logger
    except Exception as e:
        exception = e
        completion_status = 'Failure'
        raise
    finally:
        end_time = datetime.utcnow()
        duration_ms = round((end_time - start_time).total_seconds() * 1000, 2)

        custom_dimensions['completionStatus'] = completion_status
        custom_dimensions['durationMs'] = duration_ms
        message = '{} | ActivityCompleted: Activity={}, HowEnded={}, Duration={} [ms], Info = {}'.format(
            start_time, activity_name, completion_status, duration_ms, repr(activity_info))
        if exception:
            message += ', Exception={}; {}'.format(type(exception).__name__, str(exception))
            logger.error(message)
        else:
            logger.info(message)


def _get_package_version(package_name):
    import pkg_resources
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        # Azure CLI exception loads azureml-* package in a special way which makes get_distribution not working
        try:
            all_packages = pkg_resources.AvailableDistributions()  # scan sys.path
            for name in all_packages:
                if name == package_name:
                    return all_packages[name][0].version
        except Exception:
            # In case this approach is not working neither
            return None
