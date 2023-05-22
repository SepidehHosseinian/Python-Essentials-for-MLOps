# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os

from azureml._async import WorkerPool
from azureml._history.utils.constants import (OUTPUTS_DIR, DRIVER_LOG_NAME, LOGS_AZUREML_DIR,
                                              AZUREML_LOGS, AZUREML_LOG_FILE_NAME)
from azureml._history.utils.context_managers import (AmbientAuthCM, LoggedExitStack, SendRunKillSignal)
from azureml._logging.logged_lock import LoggedLock

EXECUTION_ENV_FRAMEWORK = "AZUREML_FRAMEWORK"
EXECUTION_ENV_COMMUNICATOR = "AZUREML_COMMUNICATOR"
EXECUTION_ENV_LOG_MSREST = "AZUREML_LOG_NETWORK_TRACES"
PY_SPARK_FRAMEWORK = "PySpark"
TARGET_TYPE_BATCH_AI = "batchai"
AMBIENT_AUTHENTICATION = "ambientAuthentication"
DIRECTORIES_TO_WATCH = "DirectoriesToWatch"
ONLY_IN_USER_PROCESS_FEATURES = "only_in_process_features"
DISABLE_OUTPUT_UPLOAD = "disable_output_upload"


# This logger is actually for logs happening in this file
module_logger = logging.getLogger(__name__)

AZUREML_LOG_DIR = os.environ.get("AZUREML_LOGDIRECTORY_PATH", AZUREML_LOGS)
USER_LOG_PATH = os.path.join(AZUREML_LOG_DIR, DRIVER_LOG_NAME)


# This function runs in a separate process before the user processes start.
# Only one instance of this function runs per run.
def execute_job_prep(**kwargs):
    aml_logger, azureml_log_file_path = get_azureml_logger(prefix="job_prep")
    # Log inputs to simplify debugging remote runs
    inputs = ("Execute Job Prep Inputs:: kwargs: {kwargs}").format(kwargs=kwargs)
    aml_logger.debug(inputs)

    directories_to_watch = kwargs.pop(DIRECTORIES_TO_WATCH, None)
    directories_to_watch = directories_to_watch if directories_to_watch is not None else []
    directories_to_watch = directories_to_watch if isinstance(directories_to_watch, list) else [directories_to_watch]
    context_managers = []

    if directories_to_watch:
        worker_pool = WorkerPool(_ident="HistoryTrackingWorkerPool", _parent_logger=aml_logger)
        # Enters the worker pool so that it's the last flushed element
        context_managers.append(worker_pool)
        run = _get_run_for_context_managers(worker_pool)
        context_managers.append(run._context_manager.get_content_uploader(directories_to_watch,
                                                                          azureml_log_file_path=azureml_log_file_path))

    # create OUTPUTS directory
    os.environ["AZUREML_OUTPUT_DIRECTORY"] = OUTPUTS_DIR
    if not os.path.exists(OUTPUTS_DIR):
        os.mkdir(OUTPUTS_DIR)

    if context_managers:
        logged_exit_stack = LoggedExitStack(aml_logger, context_managers)
        # Upload the ./outputs folder and any extras we need
        with logged_exit_stack:
            aml_logger.debug("Executing job prep.")


# This function runs in a separate process after all user processes exit.
# Only one instance of this function runs per run.
def execute_job_release(track_folders=None, deny_list=None, **kwargs):
    aml_logger, azureml_log_file_path = get_azureml_logger(prefix="job_release")
    directories_to_watch = kwargs.pop(DIRECTORIES_TO_WATCH, None)

    directories_to_watch = directories_to_watch if directories_to_watch is not None else []
    directories_to_watch = directories_to_watch if isinstance(directories_to_watch, list) else [directories_to_watch]

    # Log inputs to simplify debugging remote runs
    inputs = ("Execute Job Release Inputs:: kwargs: {kwargs}, "
              "track_folders: {track_folders}, "
              "deny_list: {deny_list}, "
              "directories_to_watch: {directories_to_watch}").format(kwargs=kwargs,
                                                                     track_folders=track_folders,
                                                                     deny_list=deny_list,
                                                                     directories_to_watch=directories_to_watch)
    aml_logger.debug(inputs)
    disable_output_upload = kwargs.pop(DISABLE_OUTPUT_UPLOAD, False)

    # Load run and related context managers
    py_wd_cm = get_py_wd()
    context_managers = []
    worker_pool = WorkerPool(_ident="HistoryTrackingWorkerPool", _parent_logger=aml_logger)
    # Enters the worker pool so that it's the last flushed element
    context_managers.append(worker_pool)

    # Important! decompose here so we don't construct a run object
    track_folders = track_folders if track_folders is not None else []
    if not disable_output_upload:
        track_folders = track_folders + [OUTPUTS_DIR]
    deny_list = deny_list if deny_list is not None else []

    run = _get_run_for_context_managers(worker_pool, py_wd=py_wd_cm, outputs=track_folders,
                                        deny_list=deny_list + [USER_LOG_PATH])
    if directories_to_watch:
        context_managers.append(run._context_manager.get_content_uploader(directories_to_watch,
                                                                          azureml_log_file_path=azureml_log_file_path))
    context_managers.append(run._client.output_file_tracker)
    logged_exit_stack = LoggedExitStack(aml_logger, context_managers)
    # Upload the ./outputs folder and any extras we need
    with logged_exit_stack:
        aml_logger.debug("Executing history release.")


# To ensure backward compatibility with older versions of SDK which did not support
# multi-instance run history context manager, this function will be used by Execution Service
# to switch from a single-instance to a multi-instance context manager. Over time, when there
# is no longer a concern about backward compatibility this function can be retired (following
# the retirement of its usage in Execution service). Do not change the return value to False.
def is_multi_instance_supported():
    return True


# The context manager returned by this function runs in the user process
# with the context manager wrapping the user code.
# The context manager returned by this function runs in EVERY user process in a multiprocess run.
def get_history_context(callback, args, module_logger, track_folders=None, deny_list=None, **kwargs):
    return get_history_context_manager(track_folders=track_folders, deny_list=deny_list, **kwargs)


# The context manager returned by this function runs in the user process
# with the context manager wrapping the user code.
# The context manager returned by this function runs in EVERY user process in a multiprocess run.
def get_history_context_manager(track_folders=None, deny_list=None, **kwargs):
    aml_logger, azureml_log_file_path = get_azureml_logger(get_process_name())
    directories_to_watch = kwargs.pop(DIRECTORIES_TO_WATCH, None)

    directories_to_watch = directories_to_watch if directories_to_watch is not None else []
    directories_to_watch = directories_to_watch if isinstance(directories_to_watch, list) else [directories_to_watch]
    if not is_batchai():
        if not os.path.exists(LOGS_AZUREML_DIR):
            umask = os.umask(0)
            try:
                os.makedirs(LOGS_AZUREML_DIR, exist_ok=True)
            finally:
                os.umask(umask)
        directories_to_watch.append(LOGS_AZUREML_DIR)

    # Log inputs to simplify debugging remote runs
    inputs = ("Inputs:: kwargs: {kwargs}, "
              "track_folders: {track_folders}, "
              "deny_list: {deny_list}, "
              "directories_to_watch: {directories_to_watch}").format(kwargs=kwargs,
                                                                     track_folders=track_folders,
                                                                     deny_list=deny_list,
                                                                     directories_to_watch=directories_to_watch)

    aml_logger.debug(inputs)

    only_in_process_features = kwargs.pop(ONLY_IN_USER_PROCESS_FEATURES, False)
    track_folders = track_folders if track_folders is not None else []
    deny_list = deny_list if deny_list is not None else []

    if not only_in_process_features:
        os.environ["AZUREML_OUTPUT_DIRECTORY"] = OUTPUTS_DIR
        if not os.path.exists(OUTPUTS_DIR):
            os.mkdir(OUTPUTS_DIR)

    context_managers = []

    # Load run and related context managers
    py_wd_cm = get_py_wd()

    worker_pool = WorkerPool(_ident="HistoryTrackingWorkerPool", _parent_logger=aml_logger)
    # Enters the worker pool so that it's the last flushed element
    context_managers.append(worker_pool)

    # Send the "about to be killed" signal so Runs can clean up quickly
    send_kill_signal = not os.environ.get("AZUREML_DISABLE_RUN_KILL_SIGNAL")
    kill_signal_timeout = float(os.environ.get("AZUREML_RUN_KILL_SIGNAL_TIMEOUT_SEC", "300"))
    context_managers.append(SendRunKillSignal(send_kill_signal, kill_signal_timeout))

    run = _get_run_for_context_managers(worker_pool, py_wd=py_wd_cm, outputs=track_folders + [OUTPUTS_DIR],
                                        deny_list=deny_list + [USER_LOG_PATH])
    run_context_manager = run._context_manager
    run_client = run._client

    # Prepare MLflow integration if supported
    try:
        # mlflow import is needed for registering azureml tracking loaders in mlflow
        import mlflow
        try:
            # check if the mlflow features are enabled
            from azureml.mlflow import _setup_remote
        except ImportError:
            aml_logger.warning("Could not import azureml.mlflow mlflow APIs "
                               "checking if azureml.contrib.mlflow is available.")
            import azureml.contrib.mlflow
            if "_setup_remote" not in dir(azureml.contrib.mlflow):
                # To preserve backward compatibality, add the impl of _setup_remote in case of
                # running old version of azureml.contrib.mlflow pkg with new azureml.core
                def _setup_remote(run):
                    tracking_uri = run.experiment.workspace.get_mlflow_tracking_uri() + "&is-remote=True"
                    mlflow.set_tracking_uri(tracking_uri)
                    try:
                        from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR
                    except ImportError:
                        from azureml.mlflow._internal.store import _VERSION_WARNING
                        aml_logger.warning(_VERSION_WARNING.format("_TRACKING_URI_ENV_VAR from "
                                                                   + "mlflow.tracking._tracking_service.utils"))
                        from mlflow.tracking.utils import _TRACKING_URI_ENV_VAR
                    from mlflow.tracking.fluent import _RUN_ID_ENV_VAR
                    os.environ[_TRACKING_URI_ENV_VAR] = tracking_uri
                    os.environ[_RUN_ID_ENV_VAR] = run.id
                    mlflow.set_experiment(run.experiment.name)

                    from mlflow.entities import SourceType
                    mlflow_tags = {}
                    mlflow_source_type_key = 'mlflow.source.type'
                    if mlflow_source_type_key not in run.tags:
                        mlflow_tags[mlflow_source_type_key] = SourceType.to_string(SourceType.JOB),
                    mlflow_source_name_key = 'mlflow.source.name'
                    if mlflow_source_name_key not in run.tags:
                        mlflow_tags[mlflow_source_name_key] = run.get_details()['runDefinition']['script']
                    run.set_tags(mlflow_tags)
    except ImportError:
        aml_logger.warning("Could not import azureml.mlflow or azureml.contrib.mlflow mlflow APIs "
                           "will not run against AzureML services.  Add azureml-mlflow as a conda "
                           "dependency for the run if this behavior is desired")
    else:
        aml_logger.debug("Installed with mlflow version {}.".format(mlflow.version.VERSION))
        _setup_remote(run)

    # Send heartbeats if enabled
    if run_context_manager.heartbeat_enabled:
        context_managers.append(run_context_manager.heartbeat_context_manager)

    # Catch sys.exit(0) - like signals from examples such as TF to avoid failing the run
    # Also set Run Errors for user failures
    context_managers.append(run_context_manager.status_context_manager)

    # Upload run metrics to cloud
    context_managers.append(run_client.metrics)

    # TODO uncomment after fixed spark bug
    # from azureml._history.utils.daemon import ResourceMonitor
    # context_managers.append(ResourceMonitor("ResourceMonitor", aml_logger))

    if directories_to_watch:
        # Tail the directories_to_watch to cloud
        context_managers.append(run_context_manager.get_content_uploader(directories_to_watch,
                                                                         azureml_log_file_path=azureml_log_file_path))

    # Upload the ./outputs folder and any extras we need
    if not only_in_process_features:
        context_managers.append(run_client.output_file_tracker)

    # Arrange for ambient authentication when the user specifies a method.
    ambient_auth_config = kwargs.pop(AMBIENT_AUTHENTICATION, None)
    if ambient_auth_config:
        context_managers.append(AmbientAuthCM(ambient_auth_config, run))

    # python working directory context manager is added last to ensure the
    # working directory before and after the user code is the same for all
    # the subsequent context managers
    return LoggedExitStack(aml_logger, context_managers + [py_wd_cm])


class PythonWorkingDirectory(object):
    _python_working_directory = None
    _initialization_lock = LoggedLock()

    @classmethod
    def get(cls):
        logger = module_logger.getChild(cls.__name__)
        with cls._initialization_lock:
            if cls._python_working_directory is None:
                fs_list = []
                from azureml._history.utils.filesystem import PythonFS
                py_fs = PythonFS('pyfs', logger)
                fs_list.append(py_fs)
                target_type = str(os.environ.get("AZUREML_TARGET_TYPE")).lower()
                logger.debug("Execution target type: {0}".format(target_type))
                try:
                    from pyspark import SparkContext
                    logger.debug("PySpark found in environment.")

                    if SparkContext._active_spark_context is not None:
                        logger.debug("Adding SparkDFS")
                        from azureml._history.utils.filesystem import SparkDFS
                        spark_dfs = SparkDFS("spark_dfs", logger)
                        fs_list.append(spark_dfs)
                        logger.debug("Added SparkDFS")

                    else:
                        if target_type == PY_SPARK_FRAMEWORK:
                            logger.warning("No active spark context with target type {}".format(target_type))

                except ImportError as import_error:
                    logger.debug("Failed to import pyspark with error: {}".format(import_error))

                from azureml._history.utils.context_managers import WorkingDirectoryCM
                cls._python_working_directory = WorkingDirectoryCM(logger, fs_list)

        return cls._python_working_directory


def get_py_wd():
    return PythonWorkingDirectory.get()


def is_batchai():
    target_type = os.environ.get("AZUREML_TARGET_TYPE")
    return target_type is not None and target_type.lower() == TARGET_TYPE_BATCH_AI


def get_process_name():
    distributed_process_name = "{}{}".format(os.environ.get("AZ_BATCHAI_TASK_TYPE", ""),
                                             os.environ.get("AZ_BATCHAI_TASK_INDEX", ""))
    return os.getpid() if not distributed_process_name else "{}_{}".format(distributed_process_name, os.getpid())


def get_azureml_logger(process_name=None, prefix=None):
    # Configure logging for azureml namespace - debug logs+
    aml_logger = logging.getLogger('azureml')
    aml_logger.debug("Called azureml._history.utils.context_managers.get_history_context")
    # load the msrest logger to log requests and responses
    msrest_logger = logging.getLogger("msrest")

    include_msrest = os.environ.get(EXECUTION_ENV_LOG_MSREST, None)
    aml_loggers = [aml_logger, msrest_logger] if include_msrest else [aml_logger]
    umask = os.umask(0)
    try:
        if not os.path.exists(LOGS_AZUREML_DIR):
            os.makedirs(LOGS_AZUREML_DIR, exist_ok=True)

        if not os.path.exists(AZUREML_LOG_DIR):
            os.makedirs(AZUREML_LOG_DIR, exist_ok=True)
    finally:
        os.umask(umask)

    log_name = "{}_{}".format(process_name, AZUREML_LOG_FILE_NAME) if process_name else AZUREML_LOG_FILE_NAME
    log_name = "{}_{}".format(prefix, log_name) if prefix is not None else log_name
    azureml_log_file_path = os.path.join(LOGS_AZUREML_DIR, log_name)

    # Configure loggers to: log to known file, format logs, log at specified level
    # Send it to the tracked log folder
    file_handler = logging.FileHandler(azureml_log_file_path)
    file_handler.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
    file_handler.setFormatter(formatter)

    # Also move this to RunConfig resolver
    LOG_LEVEL = int(os.environ.get("AZUREML_LOG_LEVEL", logging.DEBUG))

    for logger in aml_loggers:
        logger.setLevel(LOG_LEVEL)

        # This is not a great thing, but both revo and jupyter appear to add
        # root streamhandlers, causing too much information to be sent to the
        # user
        logger.propagate = 0

        logger.addHandler(file_handler)
    # Done configuring loggers
    return aml_logger, azureml_log_file_path


def _get_run_for_context_managers(worker_pool, py_wd=None, outputs=None, deny_list=None):
    # Important! decompose here so we don't construct a run object
    from azureml.core.run import Run
    run = Run.get_context(_worker_pool=worker_pool, allow_offline=False, used_for_context_manager=True,
                          py_wd=py_wd, outputs=outputs, deny_list=deny_list)
    return run
