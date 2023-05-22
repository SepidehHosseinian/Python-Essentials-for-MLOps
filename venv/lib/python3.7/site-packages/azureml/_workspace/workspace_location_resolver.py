# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


def get_workspace_dependent_resource_location(location):
    if location == "centraluseuap":
        return "eastus"
    return location
