# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os
logger = logging.getLogger('azure.core')
logger.setLevel(os.environ.get("AZURE_CORE_LOGLEVEL", "WARNING"))
