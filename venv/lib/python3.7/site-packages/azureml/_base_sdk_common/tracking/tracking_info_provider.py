# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Track a specific type of contextual information."""

try:
    from abc import ABCMeta

    ABC = ABCMeta('ABC', (), {})
except ImportError:
    from abc import ABC

from abc import abstractmethod


class TrackingInfoProvider(ABC):
    @abstractmethod
    def gather(self):
        """
        Gather tracking information from the local environment.

        :return: A dictionary of property/value pairs.
        :rtype: dict
        """
        return {}
