# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from .git_tracking_info_provider import GitTrackingInfoProvider
from .tracking_info_provider import TrackingInfoProvider
from .tracking_info_registry import TrackingInfoRegistry

global_tracking_info_registry = TrackingInfoRegistry()
global_tracking_info_registry.register(GitTrackingInfoProvider())

__all__ = [
    'GitTrackingInfoProvider',
    'TrackingInfoProvider',
    'TrackingInfoRegistry',
    'global_tracking_info_registry',
]
