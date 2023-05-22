# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._base_sdk_common import __version__ as VERSION
from ._tracer_factory import get_tracer

__version__ = VERSION

__all__ = [
    'get_tracer'
]
