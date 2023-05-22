# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.exceptions import AzureMLException
from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import InvalidOutputStream
from azureml._history.utils.context_managers import (LoggedExitStack, ContentUploader,
                                                     RedirectUserOutputStreams)
from azureml._async.daemon import Daemon
from azureml._history.utils.constants import USER_LOG_FILE, LOGS_DIR
from azureml._restclient.constants import RUN_ORIGIN
from azureml._logging import ChainedIdentity

import logging
import os

module_logger = logging.getLogger(__name__)


class BaseRunContext(ChainedIdentity):
    def __init__(self, run, **kwargs):
        super(BaseRunContext, self).__init__(**kwargs)
        self.run = run

    def __enter__(self):
        self._logger.debug("[START]")
        return self


class RunStatusContext(BaseRunContext):
    def __exit__(self, exc_type, exc_val, exc_tb):
        from azureml.core.run import _SubmittedRun

        set_status = not isinstance(self.run, _SubmittedRun)

        if exc_type is None and exc_val is None and exc_tb is None:
            # No Exception - flush, and if it's not a submitted run, set status to complete
            self.run._client.flush()
            self.run._client.set_completed_status(_set_status=set_status)

        # SysExit() and SysExit(0) mean everything's fine
        if exc_type == SystemExit and (exc_val.code is None or exc_val.code == 0):
            # Suppress
            return True

        # Real exception to handle
        if isinstance(exc_val, BaseException):
            self.run._client.set_failed_status(_set_status=set_status, error_details=exc_val)


class RunHeartBeatContext(BaseRunContext):
    DEFAULT_INTERVAL_SEC = 30

    def __init__(self, run, interval_sec=DEFAULT_INTERVAL_SEC, **kwargs):
        super(RunHeartBeatContext, self).__init__(run, **kwargs)
        buffered_interval_sec = RunHeartBeatContext.buffered_interval(interval_sec)
        self._daemon = Daemon(work_func=run._heartbeat,
                              interval_sec=buffered_interval_sec,
                              _ident="RunHeartBeat",
                              _parent_logger=self._logger)

    @classmethod
    def buffered_interval(cls, interval_sec):
        return 4 * interval_sec / 5

    def __enter__(self):
        self._logger.debug("[START]")
        self._daemon.start()
        return self

    def __exit__(self, exit_type, value, traceback):
        self._logger.debug("[STOP]")
        self._daemon.stop()


class RunContextManager(BaseRunContext):
    def __init__(self, run, logs=None, heartbeat_enabled=True, py_wd=None,
                 redirect_output_stream=True, **kwargs):
        super(RunContextManager, self).__init__(run, **kwargs)
        self._status_context_manager = RunStatusContext(run, _parent_logger=self._logger)

        context_managers = [run._client, self._status_context_manager]

        self.heartbeat_enabled = heartbeat_enabled
        if self.heartbeat_enabled:
            self._heartbeat_context_manager = RunHeartBeatContext(run, _parent_logger=self._logger)
            context_managers.append(self._heartbeat_context_manager)

        logs = [] if logs is None else logs
        logs = logs if isinstance(logs, list) else [logs]
        logs.append(LOGS_DIR)

        self._logger.debug("Valid logs dir, setting up content loader")
        context_managers.append(self.get_content_uploader(logs))

        if redirect_output_stream:
            user_log_path = os.path.abspath(os.path.join(LOGS_DIR, USER_LOG_FILE))
            self._redirect_output_streams_context_manager = RedirectUserOutputStreams(logger=self._logger,
                                                                                      user_log_path=user_log_path)
            context_managers.append(self._redirect_output_streams_context_manager)

        # python workingdirectory is last to preserve the original working directory
        self.context_manager = LoggedExitStack(self._logger, context_managers + [py_wd])

    def __enter__(self):
        self._logger.debug("Entered {}".format(self.__class__.__name__))
        return self.context_manager.__enter__()

    def __exit__(self, exit_type, value, traceback):
        self._logger.debug("Exited {}".format(self.__class__.__name__))
        return self.context_manager.__exit__(exit_type, value, traceback)

    @property
    def status_context_manager(self):
        return self._status_context_manager

    @property
    def output_file_context_manager(self):
        return self._output_file_context_manager

    @property
    def heartbeat_context_manager(self):
        return self._heartbeat_context_manager

    @property
    def redirect_output_streams_context_manager(self):
        try:
            return self._redirect_output_streams_context_manager
        except AttributeError:
            azureml_error = AzureMLError.create(
                InvalidOutputStream
            )
            raise AzureMLException._with_error(azureml_error)

    def get_content_uploader(self, directories_to_watch, **kwargs):
        return ContentUploader(RUN_ORIGIN,
                               self.run._client._data_container_id,
                               self.run._client.artifacts,
                               directories_to_watch=directories_to_watch,
                               **kwargs)
