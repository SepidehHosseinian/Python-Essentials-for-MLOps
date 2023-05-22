# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import time
import threading
from .chained_identity import ChainedIdentity

ACQUIRE_DURATION_THRESHOLD = 0.1


class LoggedLock(ChainedIdentity):
    def __init__(self, **kwargs):
        super(LoggedLock, self).__init__(**kwargs)
        self._lock = threading.Lock()

    def acquire(self):
        start_time = time.time()
        self._lock.acquire()
        duration = time.time() - start_time
        if duration > ACQUIRE_DURATION_THRESHOLD:
            self._logger.debug("{} acquired lock in {} s.".format(type(self).__name__, duration))

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()
