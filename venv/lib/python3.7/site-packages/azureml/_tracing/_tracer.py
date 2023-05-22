# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os

from ._constants import USER_FACING_NAME, TRACEPARENT_ENV_VAR
from ._span import Span
from ._vendored import _execution_context as execution_context


class AmlTracer:
    def __init__(self, span_processors):
        self._span_processors = span_processors

    def start_as_current_span(self, name, parent=None, user_facing_name=None):
        parent = parent or self.__class__._get_ambient_parent()
        span = Span(name, parent, self._span_processors)
        self.__class__.decorate_span(span, user_facing_name)
        span.__enter__()
        return span

    def start_span(self, name, parent=None, user_facing_name=None):
        span = Span(name, parent, self._span_processors)
        self.__class__.decorate_span(span, user_facing_name)
        return span

    @staticmethod
    def decorate_span(span, user_facing_name):
        if user_facing_name:
            span.attributes[USER_FACING_NAME] = user_facing_name

    @staticmethod
    def _get_ambient_parent():
        current_parent = execution_context.get_current_span()
        if current_parent:
            return current_parent
        traceparent = os.environ.get(TRACEPARENT_ENV_VAR, '').split('-')
        if not traceparent or len(traceparent) != 4:
            return None
        return Span._from_traceparent(*traceparent)


class DefaultTraceProvider:
    def __init__(self, tracer):
        self._tracer = tracer

    def get_tracer(self, name):
        return self._tracer

    def get_current_span(self):
        return execution_context.get_current_span()
