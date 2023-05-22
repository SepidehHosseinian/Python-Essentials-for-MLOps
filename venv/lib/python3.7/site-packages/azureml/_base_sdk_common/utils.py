# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import contextlib
import logging
import operator
import os
import requests

from collections.abc import Iterable
from os import path
from requests.adapters import HTTPAdapter
from urllib3 import Retry

module_logger = logging.getLogger(__name__)


def convert_dict_to_list(input_dict):
    """Flattens a Dictionary into a List of items."""
    output_list = []
    if not input_dict:
        return output_list

    for key, value in input_dict.items():
        output_list.append(key)
        if not isinstance(value, str) and isinstance(value, Iterable):
            for item in value:
                if isinstance(item, Iterable):
                    output_list.append(str(item))
                else:
                    output_list.append(item)
        else:
            output_list.append(value)
    return output_list


def convert_list_to_dict(input_list):
    """Converts a list of items into a Dictionary."""
    output_dict = {}
    if not input_list:
        return output_dict

    output_dict = dict(zip(input_list[::2], input_list[1::2]))
    return output_dict


def merge_dict(dict1, dict2):
    """
    merge two Dictionary and dict2 will overwrite dict1
    """
    merged_dict = {}
    dict1_keys = []
    if dict1:
        for key in dict1.keys():
            merged_dict[key] = dict1[key]
        dict1_keys = dict1.keys()
    if dict2:
        for key in dict2.keys():
            if key in dict1_keys:
                module_logger.info(
                    "merge_dict: Key {0} existed in the dict2".format(key))
            merged_dict[key] = dict2[key]

    return merged_dict


def merge_list(list1, list2, remove_duplicate=False):
    """
    merge two lists
    when remove_duplicate is false, duplicate item will be kept.
    when remove_dupliate is true, no duplicate item is allowed
    """
    merged_list = []
    if not list1 and list2:
        merged_list.extend(list2)
    elif list1 and not list2:
        merged_list.extend(list1)
    elif list1 and list2:
        for item in list1:
            merged_list.append(item)
        for item in list2:
            if (not remove_duplicate or item not in list1):
                merged_list.append(item)

    return merged_list


def list_remove_empty_items(input_list):
    # Include list items with value 0.
    return [item for item in input_list if item is not None and item != ""]


@contextlib.contextmanager
def working_directory_context(new_dir):
    """Change the working directory inside a `with' context."""
    old_dir = os.getcwd()

    if os.path.isdir(new_dir):
        os.chdir(new_dir)
    else:
        # 'new_dir' is actually a file, so use the directory component
        # (or don't move at all if there isn't one).
        os.chdir(os.path.dirname(new_dir) or '.')

    try:
        yield new_dir
    finally:
        os.chdir(old_dir)


def create_retry(
        retries=3,
        backoff_factor=0.3,
        status_forcelist=None,
):
    if status_forcelist is None:
        status_forcelist = [500, 502, 504, 503, 508]
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    return retry


def split_path(p):
    """
    Split paths into segments

    :param p: an path
    :type p: str
    :return: path segments
    :rtype: List[str]
    """

    segments = []
    while True:
        p, tail = path.split(p)
        if tail:
            segments.append(tail)
        else:
            segments.append(p)
            break
    return segments[::-1]


def to_unix_path(p):
    """
    Converts an absolute path into a unix path

    :param p: an absolute path
    :type p: str
    """

    if not p:
        return p

    segments = split_path(p)
    segments = list(filter(lambda _: _, segments))

    if ":" in segments[0]:
        segments[0] = segments[0].rstrip("\\/:")
        segments = [""] + segments

    return "/".join(segments)


def common_path(files):
    """
    Returns the directory common to all the files the paths point to.
    Assume avg segment length of m and number of path as n, this is O(mn)

    :param files: list of paths to files
    :type files: List[str]
    :return: the common path
    :rtype: str
    """
    if len(files) == 0:
        return ""

    if len(files) == 1:
        return path.split(files[0])[0]

    def common_len(s1, s2):
        shorter = min(len(s1), len(s2))
        for i in range(shorter):
            if s1[i] != s2[i]:
                return i
        return shorter

    common_segment = split_path(files[0])

    for path_segment in map(lambda p: split_path(p), files[1:]):
        cl = common_len(common_segment, path_segment)

        if cl == 0:
            return "/"

        common_segment = common_segment[:cl]

    return path.join(*common_segment)


def create_session_with_retry(retry=3):
    """
    Create requests.session with retry

    :type retry: int
    rtype: Response
    """
    retry_policy = get_retry_policy(num_retry=retry)

    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retry_policy))
    session.mount('http://', HTTPAdapter(max_retries=retry_policy))
    return session


def get_retry_policy(num_retry=3):
    """
    :return: Returns the msrest or requests REST client retry policy.
    :rtype: urllib3.Retry
    """
    status_forcelist = [413, 429, 500, 502, 503, 504]
    backoff_factor = 0.4
    retry_policy = Retry(
        total=num_retry,
        read=num_retry,
        connect=num_retry,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is 'too many 500 error responses',
        # which is not useful.
        raise_on_status=False
    )
    return retry_policy


def get_directory_size(path, size_limit=None, include_function=None, exclude_function=None):
    """
    Get the size of the directory. If size_limit is specified, stop after reaching this value.

    :type path: str
    :type include_function: Callable
    :type exclude_function: Callable

    :rtype: int
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            full_path = os.path.normpath(os.path.join(dirpath, name))

            if ((not exclude_function and not include_function)
               or (exclude_function and not exclude_function(full_path))
               or (include_function and include_function(full_path))):
                total_size += os.path.getsize(full_path)
                if size_limit and total_size > size_limit:
                    return total_size

    return total_size


def accumulate(iterable, func=operator.add):
    # https://docs.python.org/dev/library/itertools.html#itertools.accumulate
    'Return running totals'
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total
