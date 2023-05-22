# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os

import azureml._project.file_utilities as file_utilities


class BaseMapping(object):
    def __init__(self, filename):
        self.filepath = os.path.join(file_utilities.get_home_settings_directory(), filename)

    def get_values(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as fo:
                    return json.load(fo)
            except Exception:
                pass
        return None

    def add(self, key, value=None):
        values = None
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as fo:
                    values = json.load(fo)
            except Exception:
                pass

        values = self._add(values, key, value)
        with open(self.filepath, 'w') as fo:
            fo.write(json.dumps(values))

    def get(self, key):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as fo:
                    values = json.load(fo)
                    return self._get(values, key)
            except Exception:
                pass
        return None

    def delete(self, key):
        values = None
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as fo:
                    values = json.load(fo)
            except Exception:
                pass

        values = self._delete(values, key)
        with open(self.filepath, 'w') as fo:
            fo.write(json.dumps(values))

    def _get(self, values, key):
        raise NotImplementedError()

    def _add(self, values, key, value):
        raise NotImplementedError()

    def _delete(self, values, key):
        raise NotImplementedError()


class RepoKeys(BaseMapping):
    def __init__(self):
        super(RepoKeys, self).__init__("RepoKeys.json")

    def _add(self, values, key, value):
        if values is None:
            values = {}
        values[key] = value
        return values

    def _get(self, values, key):
        return values.get(key)

    def _delete(self, values, key):
        if values is None:
            return {}
        values.pop(key, None)
        return values


class ProjectMappings(BaseMapping):
    def __init__(self):
        super(ProjectMappings, self).__init__("Projects.json")

    def _add(self, values, key, value):
        if values is None:
            values = []
        values.append(key)
        return values

    def _get(self, values, key):
        if key in values:
            return key
        return None

    def _delete(self, values, key):
        if values is None:
            return []
        if key in values:
            values.remove(key)
        return values
