# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""An abstract class for run config elements."""

from abc import ABCMeta


class _AbstractRunConfigElement(object):
    """An abstract class for run config elements."""

    __metaclass__ = ABCMeta

    def __init__(self):
        """Class AbstractRunConfigElement constructor."""
        # Used for preserving the comments on load() and save()
        self._loaded_commented_map = None
        self._initialized = False

    def __setattr__(self, name, value):
        """Manage attribute validation."""
        if hasattr(self, "_initialized") and self._initialized:
            if hasattr(self, name):
                super(_AbstractRunConfigElement, self).__setattr__(name, value)
            else:
                raise AttributeError("{} has no attribute {}".format(self.__class__, name))
        else:
            # Just set it, as these are invoked from constructor before self._initialized=True
            super(_AbstractRunConfigElement, self).__setattr__(name, value)
