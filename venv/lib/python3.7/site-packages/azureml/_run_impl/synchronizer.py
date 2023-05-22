# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from requests import Session
import logging
import os

module_logger = logging.getLogger(__name__)


class Synchronizer(object):
    # Min/max refresh delay for backoff logic
    MIN_DELAY_SECS = 5
    MAX_DELAY_SECS = 90

    def __init__(self, run_object, local_root, executor, event, session=Session()):
        self._run = run_object
        self._local_root = local_root
        self._known_files = {}
        self._session = session
        self._finished = False
        self._executor = executor
        self._event = event
        os.makedirs(self._local_root, exist_ok=True)

    @staticmethod
    def get_remote_file_size(uri, session=Session()):
        resp = session.head(uri)
        resp.raise_for_status()
        return int(resp.headers['Content-Length'])

    """
    Downloads or resumes a file from a remote uri. Returns true if we actually got data.
    We would use `azure.storage.blob` here, but that doesn't support cancellation.
    So, we use the SAS URI directly.
    """
    def update_local_file_from_remote(self, local_path, uri):
        # if this file already exists, how far are we?
        start_idx = 0

        try:
            info = os.stat(local_path)
            start_idx = info.st_size
        except FileNotFoundError:
            os.makedirs(os.path.split(local_path)[0], exist_ok=True)
            pass

        remote_size = Synchronizer.get_remote_file_size(uri, self._session)
        if start_idx == remote_size:
            return False

        with self._session.get(uri,
                               headers={'Range': 'bytes={0}-'.format(start_idx)},
                               stream=True,
                               timeout=10) as resp:
            resp.raise_for_status()

            if int(resp.headers['Content-Length']) == 0:
                return False

            written = 0
            with open(local_path, "ab+") as file:
                for chunk in resp.iter_content(chunk_size=32768):
                    file.write(chunk)
                    written += 1
                    if self._event.is_set():
                        break
        return True

    def map_remote_to_local(self, remote_artifact_path):
        return os.path.join(self._local_root, remote_artifact_path)

    def state_0(self, delay=MIN_DELAY_SECS):
        key_order = sorted(self._known_files.keys())
        for remote_path, uri in [(k, self._known_files[k]) for k in key_order]:
            local_path = self.map_remote_to_local(remote_path)

            try:
                result = self.update_local_file_from_remote(local_path, uri)
                if result:
                    delay = max(self.MIN_DELAY_SECS, int(delay / 2))
                else:
                    delay = min(self.MAX_DELAY_SECS, int(1.5 * delay))
            except Exception as ex:
                # for now, throw this back on the stack, with a longer delay
                module_logger.debug("Exception updating file {} from remote:\n{}".format(local_path, ex))
                delay = min(self.MAX_DELAY_SECS, delay * 2)
                pass

        should_stop = self._finished or self._event.wait(timeout=delay)
        if should_stop:
            # we're finished here
            return

        try:
            self._executor.submit(self.state_0, delay)
        except Exception as ex:
            module_logger.debug(ex)

    def add_remote_file(self, remote_path, uri):
        if remote_path not in self._known_files:
            module_logger.debug("Found new artifact " + remote_path)
        self._known_files[remote_path] = uri

    def run_finished(self):
        self._finished = True
