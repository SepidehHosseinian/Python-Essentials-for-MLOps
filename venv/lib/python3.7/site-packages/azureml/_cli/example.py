# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from collections import namedtuple


"""Example defines an example displayed inside a command's help message.

e.g. For the following snippet, there are two Example entries named
        "Register from local folder" and "Register from GitHub url", respectively.

$ az ml module register -h
------------------------------------------------------------------
Command
    az ml module register : Create or upgrade a module.

Arguments
    --subscription-id           : Specifies the subscription Id.
    --workspace-name -w         : Workspace name.
    ... (other params) ...

Global Arguments
    --debug                     : Increase logging verbosity to show all debug logs.
    ... (other params) ...

Examples
    Register from local folder
        az ml module register --spec-file=path/to/module_spec.yaml

    Register from GitHub url
        az ml module register --spec-file=https://github.com/user/repo/path/to/module_spec.yaml
"""
# The properties 'name', 'text' to align with the help message format defined in knack:
# https://github.com/microsoft/knack/blob/master/docs/help.md#example-yaml-help
Example = namedtuple('Example', ['name', 'text'])
