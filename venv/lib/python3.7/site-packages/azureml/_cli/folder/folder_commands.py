# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os

from azureml._cli.folder.folder_subgroup import FolderSubGroup
from azureml._cli.cli_command import command
from azureml._cli import argument

from azureml.exceptions import UserErrorException


@command(
    subgroup_type=FolderSubGroup,
    command="attach",
    short_description="Attach a folder to an AzureML workspace "
                      "and optionally a specific experiment to use by default. "
                      "If experiment name is not specified, it defaults to the folder name.",
    argument_list=[
        argument.EXPERIMENT_NAME,
        argument.PROJECT_PATH
    ])
def attach_folder_to_workspace_and_experiment(
        workspace=None,
        experiment_name=None,
        path=None,
        # We should enforce a logger
        logger=None):

    path = os.path.abspath(path)
    if os.path.exists(path) and not os.path.isdir(path):
        raise UserErrorException("The provided path [{}] must be a directory".format(path))
    elif not os.path.exists(path):
        logger.info("Creating non-existent path %s", path)
        os.makedirs(path, exist_ok=True)

    logger.debug("Workspace to attach is %s", workspace._workspace_id)

    if experiment_name is None:
        path = path.rstrip('\\/')
        experiment_to_attach = os.path.basename(path)
        logger.debug("No experiment name was provided")
    else:
        experiment_to_attach = experiment_name

    logger.debug("Attaching folder %s to experiment %s", path, experiment_to_attach)
    project = workspace._initialize_folder(experiment_to_attach, directory=path)

    return project._serialize_to_dict()
