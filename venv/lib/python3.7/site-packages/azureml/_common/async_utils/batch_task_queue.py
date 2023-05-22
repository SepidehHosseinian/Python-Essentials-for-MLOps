# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from .daemon import Daemon
from .task_queue import TaskQueue
from six.moves.queue import Queue, Empty

DEFAULT_INTERVAL = 1
DEFAULT_BATCH_CUSHION = 5
DEFAULT_BATCH_SIZE = 50
MIN_BATCH_SIZE = 1


class BatchTaskQueue(TaskQueue):
    """
    A class for managing batch async operations.
    """

    def __init__(self, work_func,
                 max_batch_size=DEFAULT_BATCH_SIZE, batch_cushion=DEFAULT_BATCH_CUSHION,
                 interval=DEFAULT_INTERVAL,
                 **kwargs):
        """
        :param work_func: Work function input params of list(item), items are added with add_item
        :type work_func: function
        :param max_batch_size: The max number of elements in a batch call
        :type max_batch_size: int
        :param batch_cushion: The batch cushion between items uploaded and the specified max
        :type batch_cushion: int
        :param interval: The interval between checking and uploading added items
        :type interval: int
        """

        super(BatchTaskQueue, self).__init__(**kwargs)
        self._max_batch_size = max_batch_size
        self._batch_cushion = batch_cushion

        self._batch_size = self._max_batch_size - self._batch_cushion
        if self._batch_size <= 0:
            self._logger.warning("Batch size - batch cushion is less than 1, defaulting to 1.")
            self._batch_size = MIN_BATCH_SIZE

        self._items = Queue()
        self._work_func = work_func

        self._daemon = Daemon(self._do_work,
                              interval,
                              _parent_logger=self._logger,
                              _ident="{}Daemon".format(self.identity))
        self._daemon.start()

    def add_item(self, item):
        """
        :param func: Function to be executed asynchronously
        :type func: builtin.function
        :param task_priority: Priority for the task, higher items have higher priority
        :type task_priority: int or None
        """
        self._items.put(item)

    def _handle_batch(self):
        batch = []
        for item in range(self._batch_size):
            try:
                item = self._items.get_nowait()
                batch.append(item)
            except Empty:
                break

        self._logger.debug("Batch size {}.".format(len(batch)))
        if len(batch) > 0:
            self.add(self._work_func, batch)

    def _do_work(self):
        if not self._items.empty():
            queue_size = self._items.qsize()
            num_batches = 1 + int(queue_size / self._batch_size)
            with TaskQueue(_ident="BatchTaskQueueAdd_{}_Batches".format(num_batches)) as task_queue:
                for _ in range(num_batches):
                    task_queue.add(self._handle_batch)

    def __exit__(self, *args):
        super(BatchTaskQueue, self).__exit__(*args)
        self._daemon.stop()

    def flush(self, *args, **kwargs):
        self._do_work()
        super(BatchTaskQueue, self).flush(*args, **kwargs)
