# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor

from azureml._logging import ChainedIdentity


class WorkerPool(ChainedIdentity, ThreadPoolExecutor):
    """
    WorkerPool class, wraps futures.ThreadPoolExecutor
    """

    def __init__(self, max_workers=None, **kwargs):
        """
        :param max_workers: The maximum number of workers to create run for the pool,
                            defaults to num cores/2
        :type max_workers: int or None
        """
        super(WorkerPool, self).__init__(max_workers=max_workers, **kwargs)

    def submit(self, func, *args, **kwargs):
        self._logger.debug("submitting future: {0}".format(func.__name__))
        return super(WorkerPool, self).submit(func, *args, **kwargs)

    def shutdown(self, *args, **kwargs):
        with self._log_context("WorkerPoolShutdown"):
            super(WorkerPool, self).shutdown(*args, **kwargs)
