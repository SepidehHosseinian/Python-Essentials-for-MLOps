# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


class SnapshotDto(object):
    def __init__(self, root=None, snapshot_id=None):
        self.root = root
        self.snapshot_id = snapshot_id
