# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._base_sdk_common import __version__ as VERSION
from azureml._common.async_utils import AsyncTask
from azureml._common.async_utils import WorkerPool
from azureml._common.async_utils import TaskQueue
from azureml._common.async_utils import BatchTaskQueue
__version__ = VERSION

__all__ = ["AsyncTask", "BatchTaskQueue", "TaskQueue", "WorkerPool"]
