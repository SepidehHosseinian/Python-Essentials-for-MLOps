# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


from azureml._cli.environment.environment_subgroup import EnvironmentSubGroup
from azureml._cli.cli_command import command
from azureml._cli import argument

from azureml.core.environment import Environment

ENVIRONMENT_NAME = argument.Argument(
    "environment_name", "--name", "-n", required=True,
    help="Name of the environment")
ENVIRONMENT_VERSION = argument.Argument(
    "environment_version", "--version", "-v", required=False,
    help="Version of the environment")
ENVIRONMENT_DIRECTORY = argument.Argument(
    "environment_directory", "--directory", "-d", required=True,
    help="Directory for the environment")
ENVIRONMENT_OVERWRITE = argument.Argument(
    "environment_overwrite", "--overwrite", "", action="store_true", required=False,
    help="Overwrite any existing destination folder")


@command(
    subgroup_type=EnvironmentSubGroup,
    command="list",
    short_description="List environments in a workspace")
def list_environments(
        workspace=None,
        # We should enforce a logger
        logger=None):
    return [Environment._serialize_to_dict(x) for x in workspace.environments.values()]


@command(
    subgroup_type=EnvironmentSubGroup,
    command="show",
    short_description="Show an environment by name and optionally version",
    argument_list=[
        ENVIRONMENT_NAME,
        ENVIRONMENT_VERSION
    ])
def show_environment(
        workspace=None,
        environment_name=None,
        environment_version=None,
        # We should enforce a logger
        logger=None):
    return Environment._serialize_to_dict(Environment.get(workspace, environment_name, environment_version))


@command(
    subgroup_type=EnvironmentSubGroup,
    command="download",
    short_description="Download an environment definition to a specified directory",
    argument_list=[
        ENVIRONMENT_NAME,
        ENVIRONMENT_VERSION,
        ENVIRONMENT_DIRECTORY,
        ENVIRONMENT_OVERWRITE
    ])
def download_environment(
        workspace=None,
        environment_name=None,
        environment_version=None,
        environment_directory=None,
        environment_overwrite=None,
        # We should enforce a logger
        logger=None):
    definition = Environment.get(workspace, environment_name, environment_version)
    definition.save_to_directory(environment_directory, environment_overwrite)


@command(
    subgroup_type=EnvironmentSubGroup,
    command="register",
    short_description="Register an environment definition from a specified directory",
    argument_list=[
        ENVIRONMENT_DIRECTORY
    ])
def register_environment(
        workspace=None,
        environment_directory=None,
        # We should enforce a logger
        logger=None):
    definition = Environment.load_from_directory(environment_directory)
    result = definition.register(workspace)
    return Environment._serialize_to_dict(result)


@command(
    subgroup_type=EnvironmentSubGroup,
    command="scaffold",
    short_description="Scaffold the files for a default environment definition in the specified directory",
    argument_list=[
        ENVIRONMENT_NAME,
        ENVIRONMENT_DIRECTORY
    ])
def scaffold_environment(
        workspace=None,
        environment_name=None,
        environment_directory=None,
        # We should enforce a logger
        logger=None):
    definition = Environment(environment_name)
    definition.save_to_directory(environment_directory)
