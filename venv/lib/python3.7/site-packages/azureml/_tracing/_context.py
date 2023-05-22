# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


class Context:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id
        self.is_remote = False
        self.trace_state = {}
