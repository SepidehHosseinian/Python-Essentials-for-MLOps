# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import uuid
import six
if six.PY2:
    from contextlib2 import ContextDecorator
else:
    from contextlib import ContextDecorator


class LogScope(ContextDecorator):
    '''Convenience for logging a context'''

    START_MSG = "[START]"
    STOP_MSG = "[STOP]"

    def __init__(self, logger, name=None):
        if not name:
            name = str(uuid.uuid4())
        self._logger = logger.getChild(name)

    def __enter__(self):
        self._logger.debug(LogScope.START_MSG)
        return self._logger

    def __exit__(self, etype, value, traceback):
        if value:
            self._logger.error("{0}: {1}\n{2}".format(etype, value, traceback))
        self._logger.debug(LogScope.STOP_MSG)
