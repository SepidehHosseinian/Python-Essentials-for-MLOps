# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import random
import traceback
from datetime import datetime

from ._constants import EXCEPTION_EVENT_NAME, EXCEPTION_TYPE, EXCEPTION_MESSAGE, EXCEPTION_STACKTRACE,\
    AML_USER_ATTR_PREFIX, AML_DEV_ATTR_PREFIX
from ._context import Context
from ._event import Event
from ._status import Status, StatusCode
from ._vendored import _execution_context as execution_context


class Span:
    def __init__(self, name, parent, span_processors):
        self._name = name
        self._parent = parent
        self._trace_id = parent.trace_id if parent else generate_trace_id()
        self._span_id = generate_span_id()
        self._span_processors = span_processors
        self._start_time = None
        self._end_time = None
        self._attributes = {}
        self._events = []
        self._status = Status()
        self._w3c_traceparent = None
        self._context = None

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    @property
    def trace_id(self):
        return self._trace_id

    @property
    def span_id(self):
        return self._span_id

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def kind(self):
        return 'Internal'

    @property
    def attributes(self):
        return self._attributes

    @property
    def events(self):
        return self._events

    @property
    def status(self):
        return self._status

    def set_attribute(self, key, value):
        self._attributes[key] = value

    def set_user_facing_attribute(self, key, value):
        self._attributes['{}.{}'.format(AML_USER_ATTR_PREFIX, key)] = value

    def set_dev_facing_attribute(self, key, value):
        self._attributes['{}.{}'.format(AML_DEV_ATTR_PREFIX, key)] = value

    def add_event(self, name, attributes):
        self._events.append(Event(name, datetime.utcnow(), attributes))

    def add_exception(self, exception, additional_attributes={}):
        self._events.append(Event(
            EXCEPTION_EVENT_NAME, datetime.utcnow(), {
                **additional_attributes,
                EXCEPTION_TYPE: type(exception).__name__,
                EXCEPTION_MESSAGE: str(exception),
                EXCEPTION_STACKTRACE: traceback.format_tb(exception.__traceback__)
            }
        ))

    def start(self):
        if self._start_time is not None:
            return

        self._start_time = datetime.utcnow()

        for span_processor in self._span_processors:
            span_processor.on_start(self)

    def set_as_current(self):
        execution_context.set_current_span(self)

    def end(self):
        if self._end_time is not None:
            return

        self._end_time = datetime.utcnow()

        for span_processor in self._span_processors:
            span_processor.on_end(self)

    def set_parent_as_current(self):
        execution_context.set_current_span(self._parent)

    def get_context(self):
        if not self._context:
            self._context = Context(self._trace_id, self._span_id)
        return self._context

    def __enter__(self):
        self.set_as_current()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.set_parent_as_current()
        self._handle_exceptions(exc_type, exc_val, exc_tb)
        self.end()

    def _handle_exceptions(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return
        import traceback

        self._status = Status(StatusCode.INTERNAL)
        self._attributes[EXCEPTION_TYPE] = exc_type.__name__
        try:
            self._attributes[EXCEPTION_MESSAGE] = exc_val.message
        except AttributeError:
            self._attributes[EXCEPTION_MESSAGE] = str(exc_val)
        self._attributes[EXCEPTION_STACKTRACE] = traceback.format_tb(exc_tb)

    def to_w3c_traceparent(self):
        if not self._w3c_traceparent:
            self._w3c_traceparent = '00-{}-{}-{}'.format(
                self.trace_id.to_bytes(16, 'big').hex(),
                self.span_id.to_bytes(8, 'big').hex(),
                '01'
            )
        return self._w3c_traceparent

    @staticmethod
    def _from_traceparent(version, hex_trace_id, hex_span_span, traceflag):
        trace_id = int(hex_trace_id, 16)
        span_id = int(hex_span_span, 16)
        span = Span('', None, [])
        span._trace_id = trace_id
        span._span_id = span_id
        return span


def generate_span_id():
    return random.getrandbits(64)


def generate_trace_id():
    return random.getrandbits(128)
