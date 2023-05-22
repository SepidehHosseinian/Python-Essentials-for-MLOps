# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


from azureml._cli.experiment.experiment_subgroup import ExperimentSubGroup
from azureml._cli.cli_command import command

from azureml._restclient.workspace_client import WorkspaceClient


@command(
    subgroup_type=ExperimentSubGroup,
    command="list",
    short_description="List experiments in a workspace")
def list_experiments_in_workspace(
        workspace=None,
        # We should enforce a logger
        logger=None):

    workspace_client = WorkspaceClient(workspace.service_context)
    # TODO: Add argument.LAST_N to this command
    experiments = workspace_client.list_experiments()
    return list(experiments)
