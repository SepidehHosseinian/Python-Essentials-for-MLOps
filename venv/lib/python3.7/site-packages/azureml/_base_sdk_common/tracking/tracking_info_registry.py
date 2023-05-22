# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Track all kinds of contextual information."""

from azureml._base_sdk_common.utils import working_directory_context


class TrackingInfoRegistry(object):
    def __init__(self):
        """
        Collection of tracking info providers.

        Use :func:`register` to add providers, which should be descendants of :class:`TrackingInfoProvider`.
        """
        self.providers = []

    def gather_all(self, base_directory=None):
        """
        Gather tracking info from all registered providers.

        :param base_directory: Base directory from which to gather tracking info.
        :type base_directory; str
        :return: The gathered tracking info.
        :rtype: dict
        """
        with working_directory_context(base_directory or '.'):
            info = {}

            for provider in self.providers:
                info.update(provider.gather())

            return info

    def register(self, provider):
        """
        Register a tracking info provider.

        :param provider: The tracking info provider to register.
        :type provider: :class:`TrackingInfoProvider`
        """
        self.providers.append(provider)
