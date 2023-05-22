# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import pathspec


default_patterns = [
    ".ipynb_checkpoints",
    "azureml-logs",
    ".azureml",
    ".git",
    "outputs",
    "azureml-setup",
    "docs"
]

_aml_ignore_file_name = ".amlignore"
_git_ignore_file_name = ".gitignore"


class IgnoreFile(object):
    def __init__(self, file_path):
        """
        :param file_path: Relative path, or absolute path to the ignore file. Converted to absolute path.
        :type directory_path: str
        :type files_to_exclude: []

        :rtype: None
        """
        self._path = os.path.abspath(file_path)
        self._path_spec = None

    def _create_pathspec(self):
        if not self.exists():
            return
        with open(self._path, 'r') as fh:
            return pathspec.PathSpec.from_lines('gitwildmatch', fh)

    def get_path(self):
        """
        Get the file path.

        :rtype: str
        """
        return self._path

    def exists(self):
        """
        Checks if file exists.

        :rtype: None
        """
        return self._path and os.path.exists(self._path)

    def create_if_not_exists(self, patterns_to_exclude=default_patterns):
        """
        Creates file if it doesn't exist.

        :rtype: None
        """
        if not self.exists():
            with open(self._path, 'w') as fo:
                fo.write('\n'.join(patterns_to_exclude) + '\n')

    def is_file_excluded(self, file_path):
        """
        Checks if given file_path is excluded.

        :rtype: bool
        """

        if not self.exists():
            return False
        if not self._path_spec:
            self._path_spec = self._create_pathspec()
        if os.path.isabs(file_path):
            file_path = os.path.normpath(file_path)
            ignore_dirname = os.path.dirname(self._path)
            if len(os.path.commonprefix([file_path, ignore_dirname])) != len(ignore_dirname):
                return True
            file_path = os.path.relpath(file_path, os.path.dirname(self._path))

        return self._path_spec.match_file(file_path)


class AmlIgnoreFile(IgnoreFile):
    def __init__(self, directory_path):
        file_path = os.path.join(directory_path, _aml_ignore_file_name)
        super(AmlIgnoreFile, self).__init__(file_path)


class GitIgnoreFile(IgnoreFile):
    def __init__(self, directory_path):
        file_path = os.path.join(directory_path, _git_ignore_file_name)
        super(GitIgnoreFile, self).__init__(file_path)


def get_project_ignore_file(directory_path):
    """
    Gets the correct ignore file for the project.

    :rtype: IgnoreFile
    """
    aml_ignore = AmlIgnoreFile(directory_path)
    git_ignore = GitIgnoreFile(directory_path)

    if not aml_ignore.exists() and git_ignore.exists():
        return git_ignore
    else:
        return aml_ignore
