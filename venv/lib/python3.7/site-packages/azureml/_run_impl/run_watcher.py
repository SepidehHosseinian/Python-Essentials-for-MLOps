# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._restclient.artifacts_client import ArtifactsClient
from azureml._restclient.constants import RUN_ORIGIN, RunStatus
import os
import logging
from .synchronizer import Synchronizer

module_logger = logging.getLogger(__name__)

TERMINAL_STATES = [RunStatus.CANCELED, RunStatus.COMPLETED, RunStatus.FAILED]


class RunWatcher(object):
    # Min/max delay for run watcher backoff logic
    MIN_DELAY_SECS = 5
    MAX_DELAY_SECS = 90

    def __init__(self, run, local_root, remote_root, executor, event, session):
        self._run_object = run
        self._client = ArtifactsClient(self._run_object.experiment.workspace.service_context)
        self._artifacts_uris = {}
        self._dir_syncs = {}
        self._local_root = local_root
        self._remote_root = remote_root
        self._executor = executor
        self._event = event
        self._session = session

    def get_tb_artifacts(self, client):
        container_artifacts = client.get_artifact_by_container(RUN_ORIGIN, self._run_object._client._data_container_id)
        return filter(lambda a: a.path.startswith(self._remote_root), container_artifacts)

    """refreshes log artifacts for one pulse. returns whether run is finished"""
    def refresh_log_artifacts(self):
        for artifact in self.get_tb_artifacts(self._client):
            remote_path = artifact.path
            uri = self._client.get_file_uri(RUN_ORIGIN, self._run_object._client._data_container_id, remote_path)
            self._artifacts_uris[remote_path] = uri

            if not remote_path.startswith(self._remote_root):
                raise Exception("Remote artifact path doesn't start with known prefix")

            stripped_file_path = remote_path.replace(self._remote_root, "", 1)
            base_path = os.path.split(stripped_file_path)[0]

            # If we don't already have a synchronizer for this path, make one
            if base_path not in self._dir_syncs:
                sync = Synchronizer(run_object=self._run_object,
                                    local_root=self._local_root,
                                    executor=self._executor,
                                    event=self._event,
                                    session=self._session)
                self._dir_syncs[base_path] = sync
                self._executor.submit(sync.state_0)

            # Notify the relevant synchronizer of the new file/URI to synchronize
            this_dir_sync = self._dir_syncs[base_path]
            this_dir_sync.add_remote_file(stripped_file_path, uri)

        return self._run_object.get_status() in TERMINAL_STATES

    def refresh_requeue(self, delay=MIN_DELAY_SECS):
        try:
            if not self.refresh_log_artifacts():
                should_exit = self._event.wait(timeout=delay)
                if should_exit:
                    return
                try:
                    self._executor.submit(self.refresh_requeue)
                except RuntimeError:
                    pass
            else:
                module_logger.debug("Run {} has finished, stopping refresh of new artifacts".format(
                    self._run_object.id))
                for sync in self._dir_syncs.values():
                    sync.run_finished()

        except Exception as ex:
            module_logger.warning("Exception refreshing log artifacts for run {}: {}".format(
                self._run_object.id, ex))
            new_delay = min(delay * 2, self.MAX_DELAY_SECS)
            module_logger.debug("Retrying in {} secs.".format(new_delay))
            should_exit = self._event.wait(new_delay)
            if not should_exit:
                self._executor.submit(self.refresh_requeue, new_delay)
