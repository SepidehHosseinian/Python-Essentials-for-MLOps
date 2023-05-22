# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

try:
    from ._version import ver
    __version__ = ver
except ImportError:
    __version__ = "0.0.0+dev"

from uuid import uuid4
_ClientSessionId = str(uuid4())
