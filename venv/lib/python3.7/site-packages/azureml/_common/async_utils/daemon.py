# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import threading

from azureml._logging.chained_identity import ChainedIdentity


class Daemon(ChainedIdentity):
    '''
    Utility class for Azure ML run-tracking daemons
    '''

    def __init__(self, work_func, interval_sec, **kwargs):
        super(Daemon, self).__init__(**kwargs)
        self.interval = interval_sec
        self.kill = False
        self.current_timer = None
        self.work = work_func
        self.current_thread = None

    def _do_work(self):
        # Start timer again before work in case work is expensive
        self._reset_timer(start=True)
        self.current_thread = threading.get_ident()
        self.work()
        self.current_thread = None

    def start(self):
        '''Start a daemon (does work first). Don't call on started daemon'''
        if self.current_timer:
            raise AssertionError("Called start() on existing Timer")

        self._logger.debug("Starting daemon and triggering first instance")
        self._do_work()

    def stop(self):
        '''Stop a running daemon. Don't call on unstarted daemon'''
        with self._log_context("StoppingDaemonThread"):
            self.kill = True
            self.current_timer.cancel()

    def _reset_timer(self, start=False, interval_sec=None):
        if self.kill:
            self._logger.debug("Instructed to kill daemon. Ignoring timer reset")
            return

        if interval_sec is not None:
            self.interval = interval_sec
        self.current_timer = threading.Timer(self.interval, self._do_work)
        self.current_timer.setDaemon(True)

        if start:
            self.current_timer.start()

    def _change_interval(self, new_interval):
        self._logger.debug("Changing timer interval from {} to {}.".format(self.interval, new_interval))
        self.current_timer.cancel()
        self.interval = new_interval
        # Creates new timer
        self._do_work()

    def __str__(self):
        return 'Daemon("{0}", {1}s, work: {2})'.format(self.identity, self.interval, self.work.__name__)

    def __repr__(self):
        return str(self)
