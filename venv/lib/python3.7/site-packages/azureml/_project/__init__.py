# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import sys

from azureml._base_sdk_common import __version__ as VERSION
__version__ = VERSION

vendor_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "vendor"))
sys.path.append(vendor_folder)
