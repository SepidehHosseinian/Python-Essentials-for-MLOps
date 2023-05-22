# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os

from ._run_history_exporter import RunHistoryExporter
from ._span_processor import ExporterSpanProcessor, UserFacingSpanProcessor, AmlContextSpanProcessor, \
    AggregatedSpanProcessor
from ._tracer import AmlTracer, DefaultTraceProvider


_trace_provider = None


def get_tracer(name):
    return _get_trace_provider().get_tracer(name)


def _get_trace_provider():
    global _trace_provider
    if _trace_provider:
        return _trace_provider

    processors = []

    try:
        from azureml.telemetry import AML_INTERNAL_LOGGER_NAMESPACE
        from azureml.telemetry.logging_handler import get_appinsights_log_handler
        from azureml.telemetry import INSTRUMENTATION_KEY

        logger = logging.getLogger(AML_INTERNAL_LOGGER_NAMESPACE + __name__)
        handler = get_appinsights_log_handler(INSTRUMENTATION_KEY)
        logger.addHandler(handler)
    except Exception:
        logger = None

    if os.environ.get('AZUREML_OTEL_EXPORT_RH'):
        processors.append(AggregatedSpanProcessor([
            AmlContextSpanProcessor(UserFacingSpanProcessor(ExporterSpanProcessor(
                RunHistoryExporter(logger=logger), logger=logger
            )))
        ]))

    _trace_provider = DefaultTraceProvider(AmlTracer(processors))
    return _trace_provider
