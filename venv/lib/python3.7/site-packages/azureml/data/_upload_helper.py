# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains helper methods for dataset/datastore upload."""

import logging
import os
import re
import sys

from .dataset_error_handling import _handle_dataset_exception
from azureml._base_sdk_common.utils import to_unix_path
from azureml._common.async_utils.async_task import AsyncTask
from azureml._common.async_utils.task_queue import TaskQueue
from azureml.exceptions import UserErrorException

module_logger = logging.getLogger(__name__)
sanitize_regex = re.compile(r"^(\.*[/\\])*")


def _start_upload_task(paths_to_upload, overwrite, exists, show_progress, task_generator, continue_on_failure=True):
    # it's an estimated total because we might skip some files
    estimated_total = len(paths_to_upload)
    counter = _Counter()
    console = _get_progress_logger(show_progress, module_logger)

    console("Uploading an estimated of {} files".format(estimated_total))

    def exception_handler(e, logger):
        logger.error("Upload failed due to: ", e)
        if not continue_on_failure:
            _handle_dataset_exception(e)

    with TaskQueue(flush_timeout_seconds=float('inf'), _ident=__name__, _parent_logger=module_logger) as tq:
        for (src_file_path, target_file_path) in paths_to_upload:
            if not overwrite:
                if exists(target_file_path):
                    estimated_total -= 1
                    console("Target already exists. Skipping upload for {}".format(target_file_path))
                    continue

            task_fn = task_generator(target_file_path, src_file_path)
            future_handler = _get_task_handler(src_file_path, counter, estimated_total, show_progress,
                                               "Upload", exception_handler)
            future = tq.create_future(task_fn)
            async_task = AsyncTask(future, handler=future_handler,
                                   _ident="task_upload_{}".format(target_file_path),
                                   _parent_logger=module_logger)
            tq.add_task(async_task)

    console("Uploaded {} files".format(counter.count()))
    return counter.count()


def _get_upload_from_dir(src_path, target_path):
    src_path = src_path.rstrip("/\\")
    if not os.path.isdir(src_path):
        raise UserErrorException("Source path for upload needs to be a directory.\
             Provided path for upload is {}".format(src_path))

    paths_to_upload = []
    for dirpath, dirnames, filenames in os.walk(src_path):
        paths_to_upload += _get_upload_from_files(
            map(lambda f: os.path.join(dirpath, f), filenames),
            target_path,
            src_path,
            True)
    return paths_to_upload


def _get_upload_from_files(file_paths, target_path, relative_root, skip_root_check):
    paths_to_upload = []
    target_path = _sanitize_target_path(target_path)
    for file_path in file_paths:
        if not skip_root_check and relative_root not in file_path and relative_root != "/":
            raise UserErrorException("relative_root: '{}' is not part of the file_path: '{}'.".format(
                relative_root, file_path))
        if not os.path.isfile(file_path):
            err_msg = "'{}' does not point to a file. " + \
                "Please upload the file to cloud first if running in a cloud notebook."
            raise UserErrorException(err_msg.format(file_path))

        target_file_path = to_unix_path(file_path)
        if relative_root != "/":
            # need to do this because Windows doesn't support relpath if the partition is different
            target_file_path = os.path.relpath(target_file_path, to_unix_path(relative_root))
        else:
            # strip away / otherwise we will create a folder in the container with no name
            target_file_path = target_file_path.lstrip("/")

        if target_path:
            target_file_path = os.path.join(target_path, target_file_path)

        paths_to_upload.append((file_path, target_file_path))

    return paths_to_upload


def _get_task_handler(f, counter, total, show_progress, action, exception_handler=None):
    def handler(future, logger):
        print_progress = _get_progress_logger(show_progress, logger)

        try:
            print_progress("{}ing {}".format(action, f))
            result = future.result()
            # thanks to GIL no need to use lock here
            counter.increment()
            print_progress("{}ed {}, {} files out of an estimated total of {}".format(
                action, f, counter.count(), total))
            return result
        except Exception as e:
            if exception_handler:
                exception_handler(e, logger)
            else:
                logger.error("Task Exception", e)

    return handler


def _sanitize_target_path(target_path):
    if not target_path:
        return target_path
    return sanitize_regex.sub("", target_path)


def _get_progress_logger(show_progress, logger=None):
    console = sys.stdout

    def log(message):
        show_progress and console.write("{}\n".format(message))
        logger.info(message)

    return log


class _Counter(object):
    def __init__(self):
        self._count = 0

    def increment(self, by=1):
        self._count += by

    def count(self):
        return self._count
