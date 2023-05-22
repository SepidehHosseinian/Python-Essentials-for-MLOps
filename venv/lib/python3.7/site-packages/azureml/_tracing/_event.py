# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


class Event:
    def __init__(self, name, timestamp, attributes):
        self._name = name
        self._timestamp = timestamp
        self._attributes = attributes

    @property
    def name(self):
        return self._name

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def attributes(self):
        return self._attributes
