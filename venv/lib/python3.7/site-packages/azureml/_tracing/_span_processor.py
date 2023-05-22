# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import atexit
import logging
import os
from abc import ABC, abstractmethod
from queue import Queue, Empty
from threading import Event, Thread

from ._constants import USER_FACING_NAME, AML_DEV_ATTR_PREFIX


class SpanProcessor(ABC):
    def on_start(self, span):
        pass

    @abstractmethod
    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


class _ChainedSpanProcessor(SpanProcessor):
    def __init__(self, span_processor):
        self._next_processor = span_processor

    def on_start(self, span):
        self._next_processor.on_start(span)

    def on_end(self, span):
        self._next_processor.on_end(span)

    def shutdown(self):
        self._next_processor.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self._next_processor.force_flush(timeout_millis)


class ExporterSpanProcessor(SpanProcessor):
    def __init__(self, span_exporter, logger=None):
        self._span_exporter = span_exporter
        self._logger = logger or logging.getLogger(__name__)

    def on_end(self, span):
        try:
            self._span_exporter.export((span,))
        except Exception as e:
            self._logger.error('Exception of type {} while exporting spans.'.format(type(e).__name__))

    def shutdown(self):
        self._span_exporter.shutdown()


class UserFacingSpanProcessor(_ChainedSpanProcessor):
    def __init__(self, span_processor):
        super().__init__(span_processor)

    def on_end(self, span):
        if USER_FACING_NAME not in span.attributes:
            return

        span = _clone_span(span)
        _remove_dev_attributes(span.attributes)
        for event in span.events:
            _remove_dev_attributes(event.attributes)

        super().on_end(span)


class AmlContextSpanProcessor(_ChainedSpanProcessor):
    def __init__(self, span_processor):
        from azureml._base_sdk_common import _ClientSessionId

        super().__init__(span_processor)

        self._run_id = None
        self._session_id = _ClientSessionId

    def on_end(self, span):
        self._add_aml_context(span)
        super().on_end(span)

    def _add_aml_context(self, span):
        span.set_user_facing_attribute('session_id', self._session_id)
        span.set_user_facing_attribute('run_id', self._get_run_id())

    def _get_run_id(self):
        if not self._run_id:
            # We first check the environment variable instead of doing Run.get_context as Run.get_context can try
            # to initialize a PythonFS which in a multithreaded scenario has the potential to cause a deadlock.
            self._run_id = os.environ.get('AZUREML_RUN_ID')
            if not self._run_id:
                try:
                    from azureml.core import Run
                    self._run_id = Run.get_context().id
                except Exception:
                    self._run_id = '[Unavailable]'
        return self._run_id


class AggregatedSpanProcessor(SpanProcessor):
    def __init__(self, span_processors):
        self._span_processors = span_processors
        self._event = Event()
        self._start_queue = Queue()
        self._end_queue = Queue()
        self._start_task = None
        self._end_task = None
        if any(span_processors):
            def on_start(processor, span):
                processor.on_start(span)

            def on_end(processor, span):
                processor.on_end(span)

            self._start_task = Thread(
                target=AggregatedSpanProcessor._worker,
                args=(self._start_queue, self._event, self._span_processors, on_start)
            )
            self._end_task = Thread(
                target=AggregatedSpanProcessor._worker,
                args=(self._end_queue, self._event, self._span_processors, on_end)
            )
            self._start_task.daemon = True
            self._start_task.start()
            self._end_task.daemon = True
            self._end_task.start()
            atexit.register(self.__class__._atexit, self._event, self._end_task, self._span_processors)

    def on_end(self, span):
        self._end_queue.put(span)

    def shutdown(self):
        self.__class__._atexit(self._event, self._end_task, self._span_processors)

    def force_flush(self, timeout_millis=30000):
        all_successful = True
        for span_processor in self._span_processors:
            all_successful = span_processor.force_flush(timeout_millis) and all_successful
        return all_successful

    def __del__(self):
        self._event.set()
        if self._end_task:
            self._end_task.join(5)

    @staticmethod
    def _worker(queue, event, span_processors, action):
        while not event.is_set() or not queue.empty():
            try:
                span = queue.get(block=True, timeout=1)
                for span_processor in span_processors:
                    action(span_processor, span)
            except Empty:
                pass

    @staticmethod
    def _atexit(event, task, processors):
        event.set()
        task.join(5)
        for span_processor in processors:
            span_processor.shutdown()


def _clone_span(span):
    from ._span import Span
    from ._event import Event

    def clone_event(event):
        return Event(event.name, event.timestamp, event.attributes.copy())

    cloned = Span(span.name, span.parent, span._span_processors)
    cloned._trace_id = span.trace_id
    cloned._span_id = span.span_id
    cloned._start_time = span.start_time
    cloned._end_time = span.end_time
    cloned._attributes = span.attributes.copy()
    cloned._events = [clone_event(event) for event in span.events]
    cloned._status = span.status
    return cloned


def _remove_dev_attributes(attributes):
    for key in attributes:
        if key.startswith(AML_DEV_ATTR_PREFIX + '.'):
            del attributes[key]
