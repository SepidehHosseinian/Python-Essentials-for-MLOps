# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" cmd_computetarget.py, A file for handling compute target commands."""
# pylint: disable=line-too-long
from azureml._base_sdk_common.common import set_correlation_id
from azureml.core import remove_legacy_compute_target
from azureml.core.compute import ComputeTarget
from azureml.core.experiment import Experiment
from azureml._base_sdk_common.common import CLICommandOutput
from azureml.exceptions import ProjectSystemException
from azureml.exceptions import UserErrorException
from azureml._project.project import Project
from ._common import get_workspace_or_default

from ._common import get_cli_specific_auth, get_cli_specific_output

""" Modules """


def detach(name=None, project=None):
    """Detach compute target"""
    # Set correlation id
    set_correlation_id()

    if not project:
        project = "."

    auth = get_cli_specific_auth()
    project_object = Project(auth=auth, directory=project)
    experiment = Experiment(project_object.workspace, project_object.history.name)
    remove_legacy_compute_target(experiment, project, name)
    command_output = CLICommandOutput("Detaching {} compute target for project "
                                      "{} successful".format(name, project_object.project_directory))
    command_output.set_do_not_print_dict()

    return get_cli_specific_output(command_output)


def list_all(project=None, resource_group_name=None, workspace_name=None):
    """List all compute targets"""
    # FIXME: Table transformer in list command.
    # from azureml._base_sdk_common.data_transformers import compute_context_transformer
    # was used earlier as table transformer.
    # Set correlation id
    set_correlation_id()

    if not project:
        project = "."

    auth = get_cli_specific_auth()
    compute_targets_dict = {}
    try:
        project_object = Project(auth=auth, directory=project)

        compute_targets_dict = project_object.legacy_compute_targets
    except ProjectSystemException as e:
        if 'No cache found for current project' not in e.message:
            raise e

    compute_target_details = "\n"
    for key, value in compute_targets_dict.items():
        compute_target_details = compute_target_details + "Name={key} : Type={value}\n".format(key=key, value=value)

    registered_compute_targets = "\n"
    try:
        workspace = get_workspace_or_default(workspace_name=workspace_name, resource_group=resource_group_name,
                                             auth=auth, project_path=project)
        targets = ComputeTarget.list(workspace)
        for target in targets:
            registered_compute_targets += 'Name={} : Type={}\n'.format(target.name, target.type)
    except UserErrorException as e:
        if 'az configure --defaults' not in e.message:
            raise e

    output_str = ""
    if compute_target_details != "\n":
        output_str += "List of attached compute targets for project at {0} are: {1}\n".format(
            project_object.project_directory, compute_target_details)

    if registered_compute_targets != "\n":
        output_str += "List of compute targets registered with workspace {0} are: {1}" \
                      "".format(workspace.name, registered_compute_targets)
    if output_str == "":
        output_str = "No compute targets found"

    cli_command_output = CLICommandOutput(output_str)

    cli_command_output.set_do_not_print_dict()
    return get_cli_specific_output(cli_command_output)
