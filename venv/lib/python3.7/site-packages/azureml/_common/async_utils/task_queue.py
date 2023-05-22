# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import time

from six.moves.queue import Queue, Empty

from azureml._logging import ChainedIdentity
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import FlushTaskTimeout

from .async_task import AsyncTask
from .worker_pool import WorkerPool

# Default timeout in seconds
DEFAULT_FLUSH_TIMEOUT_SECONDS = 120


class TaskQueue(ChainedIdentity):
    """
    A class for managing async tasks.
    """

    def __init__(self, worker_pool=None, error_handler=None, flush_timeout_seconds=None, **kwargs):
        """
        :param worker_pool: Thread pool for executing tasks
        :type worker_pool: concurrent.futures.ThreadPoolExecutor
        :param error_handler: Extension point for processing error queue items
        :type error_handler: function(error, logging.Logger)
        :param timeout_seconds: Task flush timeout in seconds
        :type timeout_seconds: timeout in seconds
        """
        super(TaskQueue, self).__init__(**kwargs)
        self._tasks = Queue()
        self._results = []
        # For right now, don't need queue for errors, but it's
        # probable that we'll want the error handler looping on queue thread
        self._errors = []
        self._err_handler = error_handler
        self._worker_pool = worker_pool if worker_pool is not None else WorkerPool(_parent_logger=self._logger)
        self._task_number = 0
        self._flush_timeout_seconds = DEFAULT_FLUSH_TIMEOUT_SECONDS
        if flush_timeout_seconds:
            self._flush_timeout_seconds = flush_timeout_seconds
            self._logger.debug('Overriding default timeout to {}'.format(flush_timeout_seconds))

    def __enter__(self):
        self._logger.debug("[Start]")
        return self

    def __exit__(self, *args):
        self._logger.debug("[Stop] - waiting default timeout")
        self.flush(self.identity)

    # TODO: Adding functions with this method needs to be more configurable
    def add(self, func, *args, **kwargs):
        """
        :param func: Function to be executed asynchronously
        :type func: builtin.function
        """
        future = self.create_future(func, *args, **kwargs)
        ident = "{}_{}".format(self._tasks.qsize(), func.__name__)
        task = AsyncTask(future, _ident=ident, _parent_logger=self._logger)
        self.add_task(task)
        return task

    def add_task(self, async_task):
        """
        :param async_task: asynchronous task to be added to the queue and possibly processed
        :type async_task: azureml._async.AsyncTask
        """
        '''Blocking, no timeout add task to queue'''
        if not isinstance(async_task, AsyncTask):
            raise ValueError("Can only add AsyncTask, got {0}".format(type(async_task)))

        self._logger.debug("Adding task {0} to queue of approximate size: {1}".format(async_task.ident,
                                                                                      self._tasks.qsize()))
        self._tasks.put(async_task)

    def create_future(self, func, *args, **kwargs):
        return self._worker_pool.submit(func, *args, **kwargs)

    def flush(self, source, timeout_seconds=None):
        with self._log_context("WaitFlushSource:{}".format(source)) as log_context:

            if timeout_seconds is None:
                log_context.debug("Overriding default flush timeout from None to {}".
                                  format(self._flush_timeout_seconds))
                timeout_seconds = self._flush_timeout_seconds
            else:
                log_context.debug("flush timeout {} is different from task queue timeout {}, using flush timeout".
                                  format(timeout_seconds, self._flush_timeout_seconds))

            start_time = time.time()

            #  Take tasks off of the queue
            tasks_to_wait = []
            while True:
                try:
                    tasks_to_wait.append(self._tasks.get_nowait())
                except Empty:
                    break

            message = ""
            timeout_time = start_time + timeout_seconds

            log_context.debug("Waiting {} seconds on tasks: {}.".format(timeout_seconds, tasks_to_wait))

            not_done = True

            while not_done and time.time() <= timeout_time:
                completed_tasks = [task for task in tasks_to_wait if task.done()]
                tasks_to_wait = [task for task in tasks_to_wait if not task.done()]
                not_done = len(tasks_to_wait) != 0

                self._results.extend((task.wait(awaiter_name=self.identity) for task in completed_tasks))

                if not_done:
                    for task in tasks_to_wait:
                        message += "Waiting on task: {}.\n".format(task.ident)
                    message += "{} tasks left. Current duration of flush {} seconds.\n".format(
                        len(tasks_to_wait), time.time() - start_time)

                    time.sleep(.25)

            self._logger.debug(message)

            # Reach this case on timeout
            if not_done:
                azureml_error = AzureMLError.create(
                    FlushTaskTimeout, timeout_seconds=timeout_seconds
                )
                raise AzureMLException._with_error(azureml_error)

    @property
    def results(self):
        for result in self._results:
            yield result

    def errors(self):
        for error in self._errors:
            yield error
