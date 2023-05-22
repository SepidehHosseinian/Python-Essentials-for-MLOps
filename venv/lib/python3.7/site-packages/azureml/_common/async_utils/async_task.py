# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
This module is basically a verbose convenience around concurrent.Futures
It should be removed - it is a compatibility layer for older patterns
that were used with requests_futures
"""
from azureml._logging import ChainedIdentity


# These handlers are module-global for pickling reasons
def basic_handler(future, _):
    """Returns future's result directly with no error handling"""
    return future.result()


def noraise_handler(future, logger):
    """ Swallows (but logs) all Exceptions"""
    try:
        return future.result()
    except Exception as error:
        logger.error("Ignoring error from request: {0}".format(error))


class AsyncTask(ChainedIdentity):
    """
    An awaitable future.
    handler accepts (future, logger) and should return future.result() or raise
    """

    def __init__(self, future, handler=None, **kwargs):
        """
        :param future: future to wrap in a task
        :type future: concurrent.Future
        :param handler: Extension point for injecting code before and after future.result()
        :type handler: function(future, logging.Logger)
        """
        super(AsyncTask, self).__init__(**kwargs)
        self._future = future
        if not handler:
            self._logger.debug("Using basic handler - no exception handling")
            self._handler = basic_handler
        else:
            self._handler = handler

    @property
    def ident(self):
        return self._identity

    def wait(self, awaiter_name=None):
        """Wait until the future is done"""
        if awaiter_name is None:
            awaiter_name = '<name not provided>'
        with self._log_context("WaitingTask") as context:
            context.debug("Awaiter is {}".format(awaiter_name))
            res = self._handler(self._future, self._logger)
        return res

    def done(self):
        """Check if the task is done"""
        return self._future.done()

    def cancel(self):
        """
        https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.cancel
        """
        with self._log_context("CancellingTask"):
            cancelled = self._future.cancel()
        self._logger.debug("Canceled Task {0}: {1}".format(self._identity, cancelled))
        return cancelled

    def _internal_rep(self):
        return "AsyncTask({})".format(self._identity)

    def __repr__(self):
        return self._internal_rep()

    def __str__(self):
        return self._internal_rep()

    def __lt__(self, other):
        return self._identity < other._identity
