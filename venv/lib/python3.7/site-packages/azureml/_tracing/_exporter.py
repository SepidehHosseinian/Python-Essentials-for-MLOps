# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from abc import ABC, abstractmethod
from datetime import datetime


class SpanExporter(ABC):
    @abstractmethod
    def export(self, spans):
        pass

    @abstractmethod
    def shutdown(self):
        pass


def to_iso_8601(time):
    if isinstance(time, datetime):
        return time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return time
