# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import ctypes
import os
import shutil

from azureml._base_sdk_common.common import AZUREML_DIR


def create_directory(path, set_hidden=False):
    """
    Create a directory and subdirs if they don't exist, and optionally set directory as hidden

    :type path: str
    :type set_hidden: bool

    :rtype None
    """
    if os.path.exists(path):
        return
    os.makedirs(path)

    if set_hidden:
        make_file_or_directory_hidden(path)


def make_file_or_directory_hidden(path):
    """
    Make a file or directory hidden

    :type path: str

    :rtype str
    """
    if os.name == 'nt':
        ctypes.windll.kernel32.SetFileAttributesW(path, 0x02)
    else:
        dirname, filename = os.path.split(path)
        if filename[0] != ".":
            new_path = os.path.join(dirname, "." + filename)
            os.rename(path, new_path)
            return new_path
    return path


def get_home_settings_directory():
    """
    Returns the home directory

    :rtype str
    """
    home = os.environ.get("HOME")
    if not home and os.name == "nt":
        home = os.environ.get("USERPROFILE")
    if not home and os.name == "nt":
        home_drive = os.environ.get("HOMEDRIVE")
        home_path = os.environ.get("HOMEPATH")
        if home_drive and home_path:
            home = home_drive + home_path
    if not home:
        home = os.environ.get("LOCALAPPDATA")
    if not home:
        home = os.environ.get("TMP")
    if not home:
        raise ValueError("Cannot find HOME env variable")

    settings_dir = os.path.join(home, AZUREML_DIR)
    create_directory(settings_dir, True)
    return settings_dir


def copy_all_files_from_directory(source_directory, destination_directory):
    """
    Copies all files from source_directory into destination_directory

    :type source_directory: str
    :type destination_directory: str

    :rtype None
    """
    for filename in os.listdir(source_directory):
        source_dir_file_path = os.path.join(source_directory, filename)
        dest_dir_file_path = os.path.join(destination_directory, filename)

        if os.path.isdir(source_dir_file_path) and os.path.exists(dest_dir_file_path):
            continue

        shutil.move(source_dir_file_path, dest_dir_file_path)
