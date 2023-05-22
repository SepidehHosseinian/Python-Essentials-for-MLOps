# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import threading

from azureml._restclient.constants import RUN_ORIGIN
from azureml.exceptions import AzureMLException, AzureMLAggregatedException


class WorkingDirectoryFS(object):
    def __init__(self, ident):
        self._ident = ident

    def ident(self):
        return self._ident

    def get_abs_working_dir(self):
        raise NotImplementedError("base")

    def set_working_dir(self, absolute_path):
        raise NotImplementedError("base")


class TrackableFS(object):
    def __init__(self, ident):
        self._ident = ident

    def track(self, run_tracker, trackfolders, blacklist):
        raise NotImplementedError("base")


class PythonFS(WorkingDirectoryFS, TrackableFS):
    # Don't want multiple threads behaving poorly
    # TODO: Convert to a LogLock that auto logs in __enter/exit__
    _lock = threading.Lock()
    __initialized = False

    def __init__(self, ident, logger):
        if PythonFS.__initialized:
            raise Exception("You can't create more than 1 instance of PythonFS")
        PythonFS.__initialized = True
        super(PythonFS, self).__init__(ident)
        self.logger = logger
        self.original_py_dir = os.getcwd()

    def get_abs_working_dir(self):
        with PythonFS._lock:
            path = os.path.abspath(os.getcwd())
            self.logger.info("Current working dir: {0}".format(path))
            return path

    def set_working_dir(self, updated_absolute_path):
        with PythonFS._lock:
            current_absolute_path = os.path.abspath(os.getcwd())
            if current_absolute_path != updated_absolute_path:
                self.logger.info("Setting working dir to {0}".format(updated_absolute_path))
                if os.path.exists(updated_absolute_path):
                    os.chdir(updated_absolute_path)
                else:
                    self.logger.warning("Failed to set working dir to {0}, the "
                                        "directory does not exist.".format(updated_absolute_path))
            else:
                self.logger.info("Working dir is already updated {}".format(updated_absolute_path))

    def track(self, artifacts_client, container_id, track_folders, blacklist):
        exception_messages = []
        for path in track_folders:
            if os.path.exists(path):
                self.logger.debug("{0} exists as directory, uploading..".format(path))
                try:
                    self._upload_folder(artifacts_client, container_id, path, blacklist)
                except AzureMLException as ex:
                    if ex.message is not None:
                        exception_messages.append(ex.message)
            else:
                self.logger.debug("Skipping {0}".format(path))

        if len(exception_messages) > 0:
            raise AzureMLAggregatedException(exception_messages)

    def _upload_folder(self, artifacts_client, container_id, path, blacklist):
        with PythonFS._lock:
            for pathl, _subdirs, files in os.walk(path):
                paths_to_upload = []
                for _file in files:
                    fpath = os.path.join(pathl, _file)
                    from azureml.history._tracking import AZUREML_LOG_FILE_NAME
                    if not fpath.endswith(AZUREML_LOG_FILE_NAME) and fpath not in blacklist:
                        # fname = os.path.normpath(os.path.join(pathl, _file))
                        self.logger.debug("Found and adding path to upload: {0}".format(fpath))
                        paths_to_upload.append(fpath)
                if paths_to_upload:
                    self.logger.debug("Paths to upload is {} in dir {}".format(paths_to_upload, pathl))
                else:
                    self.logger.debug("Paths to upload is empty in dir {}".format(pathl))
                if paths_to_upload:
                    artifacts_client.upload_files(paths_to_upload,
                                                  RUN_ORIGIN,
                                                  container_id)


class SparkDFS(WorkingDirectoryFS, TrackableFS):
    _lock = threading.Lock()

    # SparkDFS created only when cluster exists
    def __init__(self, ident, logger):
        super(SparkDFS, self).__init__(ident)
        self.logger = logger
        from pyspark.sql import SparkSession

        self.spark = SparkSession.builder.getOrCreate()
        config = self.spark._sc._jsc.hadoopConfiguration()

        dfs_cwd = self.spark._sc._gateway.jvm.org.apache.hadoop.fs.Path(".")
        self.file_system = dfs_cwd.getFileSystem(config)
        self.logger.debug("SparkDFS tracking {0}".format(self.file_system))

        self.target_type = str(os.environ.get("AZUREML_TARGET_TYPE")).lower()
        if self.target_type == "cluster":  # TODO is this pyspark
            from azureml._history.utils._hdi_utils import get_hdispark_working_dir
            self.original_wasb_dir = get_hdispark_working_dir()
        else:
            self.original_wasb_dir = None

    def get_abs_working_dir(self):
        with SparkDFS._lock:
            cwd = self.file_system.getWorkingDirectory().toString()
            self.logger.debug("Running Spark job in {0}".format(cwd))
            return cwd

    def set_working_dir(self, absolute_path):
        with SparkDFS._lock:
            if self.target_type == "cluster" and self.original_wasb_dir:
                # TODO move logic to this file
                from azureml._history.utils._hdi_utils import get_hdispark_working_dir, set_hdispark_working_dir
                cd = get_hdispark_working_dir()
                self.logger.debug("Reverting HDI working dir from {0} to {1}".format(cd, self.original_wasb_dir))
                set_hdispark_working_dir(self.original_wasb_dir)
            else:
                self.logger.debug(("SparkDFS NOOP. Target type should be cluster and origin_wasb_dir"
                                   "should not be None"))

    def track(self, artifacts_client, container_id, track_folders, blacklist):
        if self.target_type == "cluster":
            self._upload_hdi_outputs(artifacts_client, container_id)

    def _upload_hdi_outputs(self, artifacts_client, container_id):
        import timeit

        from azureml._history.utils._hdi_utils import get_hdispark_working_dir
        hdi_wd = get_hdispark_working_dir()
        self.logger.info("Uploading HDI outputs for {}".format(hdi_wd))
        track_prefix = "outputs"

        is_wasb = any(hdi_wd.startswith(prefix) for prefix in ["wasb:"])
        force_upload_by_HDFS = self._parse_bool(os.environ.get("FORCE_UPLOAD_BY_HDFS", False))
        self.logger.debug("Feature flag force_upload_by_HDFS is {}".format(force_upload_by_HDFS))

        start_time = timeit.default_timer()
        if not force_upload_by_HDFS and is_wasb:
            self.logger.debug("Uploading by wasb")
            strategy = "wasb"
            from azureml._history.utils._hdi_wasb_utils import get_wasb_container_url, \
                get_regular_container_path, get_container_sas
            wasburl = get_wasb_container_url()
            container_path = get_regular_container_path(wasburl)
            from azureml._history.utils._hdi_utils import get_working_prefix
            run_prefix = get_working_prefix(hdi_wd, wasburl, track_prefix)
            container_sas = get_container_sas(wasburl)

            self.logger.debug("Tracking outputs in {0}/{1} with prefix {2}".format(container_path,
                                                                                   run_prefix,
                                                                                   track_prefix))
            artifacts_client.batch_ingest_from_sas(RUN_ORIGIN,
                                                   container_id,
                                                   container_sas,
                                                   container_path,
                                                   run_prefix,
                                                   track_prefix)
        else:
            self.logger.debug("Uploading by hdfs")
            strategy = "hdfs"
            from azureml._history.utils._hdi_utils import upload_from_hdfs
            upload_from_hdfs(artifacts_client, container_id, track_prefix)
        elapsed = timeit.default_timer() - start_time
        self.logger.debug("HDI upload by {} took {}".format(strategy, elapsed))

    def _parse_bool(self, value):
        return value.lower() == "true" if isinstance(value, str) else bool(value)
