# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains class for specifying logging detail level."""
import logging


class NewLoggingLevel(object):
    """Changes the logging level for a specified logger.

    :param logger_name: The logger whose level will be changed.
    :type logger_name: str
    :param level: The new logging level.
    :type level: int
    """

    def __init__(self, logger_name, level=logging.WARNING):
        """Class NewLoggingLevel constructor.

        :param logger_name: The logger whose level will be changed.
        :type logger_name: str
        :param level: The new logging level.
        :type level: int
        """
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.old_level = self.logger.level
        self.new_level = level

    def __enter__(self):
        """Enter context manager.

        :return: logger
        :rtype: RootLogger
        """
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, et, ev, tb):
        """Exit context manager.

        :param et: Exception type
        :type et:
        :param ev: Exception value
        :type ev:
        :param tb: Traceback
        :type tb:
        """
        self.logger.setLevel(self.old_level)
