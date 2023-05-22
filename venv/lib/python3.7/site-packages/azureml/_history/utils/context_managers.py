# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from __future__ import print_function
from azureml._history.utils.log_scope import LogScope

import errno
import os
import six
import logging
import sys
import traceback
from azureml._history.utils.constants import LOGS_AZUREML_DIR


if six.PY2:
    from contextlib2 import ExitStack
else:
    from contextlib import ExitStack


class OutputCollector(object):
    def __init__(self, stream, processor):
        self._inner = stream
        self.processor = processor

    def write(self, buf):
        self.processor(buf)
        self._inner.write(buf)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class LoggedExitStack(object):
    def __init__(self, logger, context_managers=None):
        self._logger = logger
        self._exit_stack = ExitStack()

        # TODO make this cleaner, types would be nice
        context_managers = context_managers if context_managers is not None else []

        self.context_managers = (context_managers if isinstance(context_managers, list)
                                 else [context_managers])

    def __enter__(self):
        self._exit_stack.__enter__()
        for context_manager in self.context_managers:
            self._exit_stack.enter_context(LogScope(self._logger,
                                                    context_manager.__class__.__name__))
            self._exit_stack.enter_context(context_manager)
        return self

    def __exit__(self, *args):
        return self._exit_stack.__exit__(*args)


class WorkingDirectoryCM(object):
    def __init__(self, logger, fs_list):
        ids = [fs.ident() for fs in fs_list]
        if len(ids) != len(set(ids)):
            raise Exception("Fs ids are not unique: {}".format(ids))

        self.logger = logger.getChild("workingdir")
        self.fs_list = fs_list
        self.prev_paths = {fs.ident(): None for fs in fs_list}
        self.logger.debug("Pinning working directory for filesystems: {0}".format(list(self.prev_paths.keys())))

    def __enter__(self):
        self.logger.debug("[START]")
        for fs in self.fs_list:
            path = fs.get_abs_working_dir()
            self.logger.debug("Calling {}".format(fs.ident()))
            self.logger.debug("Storing working dir for {0} as {1}".format(fs.ident(), path))
            self.prev_paths[fs.ident()] = path
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for fs in self.fs_list:
            self.logger.debug("Calling {}".format(fs.ident()))
            path = fs.get_abs_working_dir()
            old_path = self.prev_paths[fs.ident()]
            if path != old_path:
                self.logger.debug("{} has path {}".format(fs.ident(), path))
            self.logger.debug("Reverting working dir from {0} to {1}".format(path, old_path))
            fs.set_working_dir(old_path)
        self.logger.debug("[STOP]")
        return False

    def track(self, artifacts_client, container_id, track_folders, blacklist):
        self.logger.debug("Uploading tracked directories: {0}, excluding {1}".format(track_folders, blacklist))
        for fs in self.fs_list:
            self.logger.debug("Calling track for {}".format(fs.ident()))
            fs.track(artifacts_client, container_id, track_folders, blacklist)
        return True


class RedirectUserOutputStreams(object):
    def __init__(self, logger, user_log_path):
        self.user_log_path = user_log_path
        self.logger = logger

    def __enter__(self):
        self.logger.debug("Redirecting user output to {0}".format(self.user_log_path))
        user_log_directory, _ = os.path.split(self.user_log_path)
        if not os.path.exists(user_log_directory):
            try:
                os.makedirs(user_log_directory)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        self.user_log_fp = open(self.user_log_path, "at+")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = OutputCollector(sys.stdout, self.user_log_fp.write)
        sys.stderr = OutputCollector(sys.stderr, self.user_log_fp.write)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_val:
                # The default traceback.print_exc() expects a file-like object which
                # OutputCollector is not. Instead manually print the exception details
                # to the wrapped sys.stderr by using an intermediate string.
                # trace = traceback.format_tb(exc_tb)
                trace = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                print(trace, file=sys.stderr)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            self.user_log_fp.close()
            self.logger.debug("User scope execution complete.")


class TrackFolders(object):
    def __init__(self, py_wd, artifacts_client, container_id, trackfolders, deny_list):
        self.py_wd = py_wd
        self.artifacts_client = artifacts_client
        self.container_id = container_id
        self.trackfolders = trackfolders
        self.deny_list = deny_list

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.upload_tracked_files()

    def upload_tracked_files(self):
        self.py_wd.track(self.artifacts_client, self.container_id, self.trackfolders, self.deny_list)


class SendRunKillSignal(object):
    def __init__(self, send_kill_signal=True, kill_signal_timeout=40):
        self._send_signal = send_kill_signal
        self._kill_timeout = kill_signal_timeout

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._send_signal:
            # Saddest thing in the world - cyclic dependency
            from azureml._run_impl.run_base import _RunBase
            _RunBase._kill(timeout=self._kill_timeout)


class AmbientAuthCM(object):
    def __init__(self, config, run):
        config = config if isinstance(config, dict) else {}

        self.method = config.get("method")
        self.app_id = config.get("appId")
        self.secret = config.get("secret")
        self.tenant = config.get("tenant")
        self.delete_secret_after_run = config.get("deleteSecretAfterRun")

        self.run = run

    def __enter__(self):
        if self.method == "aadtoken":
            os.environ["AZUREML_AAD_TOKEN_SECRET_NAME"] = self.secret
        elif self.method == "runtoken":
            os.environ["AZUREML_RUN_TOKEN_USE_AMBIENT_AUTH"] = "true"
        elif self.method == "serviceprincipal":
            from azureml._base_sdk_common.common import perform_interactive_login
            perform_interactive_login(service_principal=True, username=self.app_id,
                                      password=self.run.get_secret(self.secret), tenant=self.tenant)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.delete_secret_after_run is True and self.method in ["aadtoken", "serviceprincipal"] and self.secret:
            self.run.experiment.workspace.get_default_keyvault().delete_secret(self.secret)


class UploadLogsCM(object):
    def __init__(self, logger, run_tracker, driver_log_name, user_log_path, azureml_log_file_path=None,
                 azureml_log_dir=None, azureml_log_suffix=None):
        self.azureml_log_file_path = azureml_log_file_path
        self.azureml_log_dir = azureml_log_dir
        self.azureml_log_suffix = azureml_log_suffix
        self.user_log_path = user_log_path
        self.driver_log_name = driver_log_name
        self.run_tracker = run_tracker
        self.logger = logger

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.debug("Uploading driver log")
        if self.driver_log_name is not None:
            self.run_tracker.upload_file(self.driver_log_name, self.user_log_path)
        self._upload_azureml_log()

    def _upload_azureml_log(self):
        azureml_logs = {}
        if self.azureml_log_dir and self.azureml_log_suffix and os.path.exists(self.azureml_log_dir):
            from os import listdir
            from os.path import isfile, join
            azureml_logs = {f: os.path.join(self.azureml_log_dir, f) for f in listdir(self.azureml_log_dir)
                            if f.endswith(self.azureml_log_suffix)
                            and isfile(join(self.azureml_log_dir, f))}
        if self.azureml_log_file_path and os.path.exists(self.azureml_log_file_path):
            azureml_logs[os.path.basename(self.azureml_log_file_path)] = self.azureml_log_file_path
        for azureml_log in azureml_logs:
            azureml_log_name_blob = os.path.join(LOGS_AZUREML_DIR, azureml_log)
            self.logger.debug("Uploading azureml.log path {}".format(azureml_logs[azureml_log]))
            self.run_tracker.upload_file(azureml_log_name_blob, azureml_logs[azureml_log])


class ContentUploader(object):
    def __init__(self, origin, container, artifacts_client, directories_to_watch,
                 parallelism=None, azureml_log_file_path=None):

        self.origin = origin
        self.container = container
        self.artifacts_client = artifacts_client
        self.directories_to_watch = directories_to_watch
        self.threads = []
        self.logger = logging.getLogger(__name__)
        self.parallelism = parallelism
        self.azureml_log_file_path = azureml_log_file_path

    def __enter__(self):
        from azureml._run_impl.file_watcher import FileWatcher

        # create a thread to watch files
        self.logger.debug("starting file watcher")

        self.file_watcher = FileWatcher(self.directories_to_watch,
                                        self.origin,
                                        self.container,
                                        self.artifacts_client,
                                        self.logger,
                                        parallelism=self.parallelism,
                                        azureml_log_file_path=self.azureml_log_file_path)
        self.file_watcher.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.debug("exiting ContentUploader, waiting for file_watcher to finish upload...")
        self.file_watcher.finish()
        self.file_watcher.join()
        self.logger.debug("file watcher exited")
