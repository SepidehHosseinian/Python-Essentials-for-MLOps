# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
A file for base_sdk_common cli_wrapper functions.
"""

import logging
import os
import sys

from azureml._cli.aml_cli import AZUREML_CLI_IN_USE
from azureml._base_sdk_common.common import AZUREML_ARM_ACCESS_TOKEN, AZUREML_SUBSCRIPTION_ID
from azureml.core.authentication import ArmTokenAuthentication, AzureCliAuthentication, InteractiveLoginAuthentication
from azureml.core.workspace import Workspace
from azureml.exceptions import AuthenticationException
from azureml.exceptions import UserErrorException
from azureml._project.project import Project

try:  # python 3
    from configparser import ConfigParser
except ImportError:  # python 2
    from ConfigParser import ConfigParser


module_logger = logging.getLogger(__name__)

# DEFAULT MACRO NAMES for az configure defaults
DEFAULT_RESOURCE_GROUP_NAME_KEY = "group"


def get_cli_specific_auth():
    """
    Returns the cli specific auth.
    For azure.cli, returns an object of azureml.core.authentication.AzureCliAuthentication
    For azureml._cli, returns an object of azureml.core.authentication.ArmTokenAuthentication
    :return:
    :rtype: azureml.core.authentication.AbstractAuthentication
    """
    if AZUREML_CLI_IN_USE:
        if AZUREML_ARM_ACCESS_TOKEN in os.environ:
            return ArmTokenAuthentication(os.environ[AZUREML_ARM_ACCESS_TOKEN])
        else:
            return InteractiveLoginAuthentication()

    # az cli case
    return AzureCliAuthentication()


def get_default_subscription_id(auth):
    """
    Returns the default subscription id based on the auth object type.
    This function could be removed later on.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :return: subscription id
    :rtype: str
    """
    if isinstance(auth, (ArmTokenAuthentication, InteractiveLoginAuthentication)):
        if AZUREML_SUBSCRIPTION_ID in os.environ:
            return os.environ[AZUREML_SUBSCRIPTION_ID]
    elif isinstance(auth, AzureCliAuthentication):
        return auth._get_default_subscription_id()

    raise AuthenticationException("The default subscription id cannot be determined for {} "
                                  "and no subscription id was specified.".format(auth.__class__.__name__))


def get_cli_specific_output(command_output):
    """
    Returns CLI specific output.
    For azure.cli, makes some prints to stdout and the returns the JSON returned by the service.
    For azureml._cli, returns a JSON object without printing anything on stdout.
    :param command_output:
    :type command_output: CLICommandOutput
    :return:
    :rtype json:
    """
    if AZUREML_CLI_IN_USE:
        # Return the full JSON in azureml._cli case.
        return command_output.get_json_dict()
    else:
        print(command_output.get_command_output())

        if not command_output.get_do_not_print_dict():
            # Returning the JSON object from service.
            return command_output.get_json_dict(exclude_command_output=True)


def cli_exception_handler(ex):
    try:
        # Wrap as a CLIError and the exception will be handled by az cli framework.
        # The framework will display the error message in red and skip the exception trace for CLIError.
        # Refer to the framework's source code for details.
        # https://github.com/Azure/azure-cli/blob/76742362/src/azure-cli-core/azure/cli/core/util.py#L58
        from knack.util import CLIError
        raise CLIError(ex)
    except ImportError:
        # If failed to import CLIError (which should never happen), use sys.exit instead.
        # Printing exception as sys.exit skips the exception trace.
        # sys.exit is okay for CLI commands, as each command is a python process, which finishes as
        # the command finishes.
        sys.exit(ex)


def get_default_property(property_name):
    """
    :return: Returns the default property value for property_name, if set using "az configure" command.
    Return None if no default value found.
    :rtype: str
    """
    config = ConfigParser()
    config_folder = ".azureml" if AZUREML_CLI_IN_USE else ".azure"
    config.read(os.path.expanduser(os.path.join('~', config_folder, 'config')))
    if not config.has_section('defaults'):
        return None

    if config.has_option('defaults', property_name):
        return config.get('defaults', property_name)
    else:
        return None


def get_workspace_or_default_name(workspace_name,
                                  throw_error=False, subscription_id=None, auth=None, project_path=None):
    """
    Order is
    1) Get workspace name from the specified parameter,
    2) From project context,
    3) Using az configure defaults.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param workspace_name:
    :type workspace_name: str
    :param throw_error: throw_error = True throws an error if eventual workspace_name=None
    :type throw_error: bool
    :return: Returns the provided or default value of the workspace name.
    """
    if workspace_name:
        return workspace_name

    project_object = _get_project_object(subscription_id=subscription_id, auth=auth, project_path=project_path)
    if project_object:
        return project_object.workspace.name

    if throw_error:
        raise UserErrorException('Error, default workspace not set and workspace name parameter not provided.'
                                 '\nPlease set a default workspace using "az ml folder attach -w myworkspace -g '
                                 'myresourcegroup" or provide a value for the workspace name parameter.')
    else:
        return workspace_name


def get_experiment_or_default_name(experiment_name,
                                   throw_error=False, subscription_id=None, auth=None, project_path=None,
                                   logger=None):
    """
    Order is
    1) Get workspace name from the specified parameter,
    2) From project context,
    3) Using az configure defaults.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param workspace_name:
    :type workspace_name: str
    :param throw_error: throw_error = True throws an error if eventual workspace_name=None
    :type throw_error: bool
    :return: Returns the provided or default value of the workspace name.
    """
    if experiment_name:
        return experiment_name

    logger.debug("Searching for config.json at %s with auth %s and subId %s",
                 project_path, type(auth).__name__, subscription_id)
    project_object = _get_project_object(subscription_id=subscription_id, auth=auth,
                                         project_path=project_path, logger=logger)
    if project_object:
        return project_object.experiment.name

    if throw_error:
        raise UserErrorException('Error, default experiment not set and experiment name parameter not provided.'
                                 '\nPlease provide a value for the experiment name '
                                 'parameter.')
    else:
        return experiment_name


def get_resource_group_or_default_name(resource_group_name,
                                       throw_error=False, subscription_id=None, auth=None, project_path=None):
    """
    Order is
    1) Get workspace name from the specified parameter,
    2) From project context,
    3) Using az configure defaults.
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param resource_group_name:
    :type resource_group_name: str
    :param throw_error: throw_error = True throws an error if eventual resource_group_name=None
    :type throw_error: bool
    :param project_path:
    :type project_path: str
    :return: Returns the provided or default value of the resource_group_name.
    """
    if not resource_group_name:
        project_object = _get_project_object(subscription_id=subscription_id, auth=auth, project_path=project_path)
        if project_object:
            return project_object.workspace.resource_group

        resource_group_name = get_default_property(DEFAULT_RESOURCE_GROUP_NAME_KEY)
        if throw_error and not resource_group_name:
            raise UserErrorException('Error, default resource group not set.\n'
                                     'Please run "az configure --defaults group=<resource group name>" '
                                     'to set default resource group,\n or provide a value for the '
                                     '--resource-group parameter.')
        else:
            return resource_group_name
    else:
        return resource_group_name


def get_workspace_or_default(
        subscription_id=None,
        resource_group=None,
        workspace_name=None,
        auth=None,
        project_path=None,
        logger=None):
    """
    Order is
    1) Get workspace from the specified parameters,
    2) From project context,
    3) Using az configure defaults.
    :param workspace_name:
    :param resource_group:
    :param auth:
    :param project_path:
    :return:
    """

    if not logger:
        logger = module_logger

    if not auth:
        auth = get_cli_specific_auth()
        logger.debug("No auth specified, using authentication {}".format(type(auth).__name__))

    if resource_group and workspace_name:
        # Simple case where both are specified. The only way to get workspace with no
        # az configure support for 'mlworkspace' is user explicitly specified parameters
        # Technically resource group can be az configured in
        if not subscription_id:
            subscription_id = get_default_subscription_id(auth)
        return Workspace(subscription_id, resource_group, workspace_name, auth=auth)

    if project_path:
        logger.debug("Project path %s set", project_path)
        try:
            return Workspace.from_config(path=project_path, auth=auth, _logger=logger)
        except UserErrorException as ex:
            if project_path != ".":
                logger.warning("The provided path %s did not contain a config.json, "
                               "falling back to CLI configuration.", project_path)

    if not subscription_id:
        subscription_id = get_default_subscription_id(auth)

    if not workspace_name:
        workspace_name = get_workspace_or_default_name(workspace_name, throw_error=True,
                                                       subscription_id=subscription_id, auth=auth,
                                                       project_path=project_path)
    if not resource_group:
        resource_group = get_resource_group_or_default_name(resource_group, throw_error=True,
                                                            subscription_id=subscription_id, auth=auth,
                                                            project_path=project_path)

    return Workspace(subscription_id, resource_group, workspace_name, auth=auth)


def _get_experiment_or_default(
        workspace=None,
        experiment_name=None,
        project_path=None,
        logger=None):

    if not logger:
        logger = module_logger

    if not experiment_name:
        logger.debug("No experiment name provided, searching for default")
        experiment_name = get_experiment_or_default_name(experiment_name, throw_error=True,
                                                         subscription_id=workspace.subscription_id,
                                                         auth=workspace._auth_object,
                                                         project_path=project_path, logger=logger)

    from azureml.core import Experiment
    return Experiment(workspace, experiment_name)


def _get_project_object(subscription_id=None, auth=None, project_path=None, logger=None):
    if not logger:
        logger = module_logger

    if not auth:
        logger.debug("_get_project_object fetching auth because none was provided")
        auth = get_cli_specific_auth()

    if not subscription_id:
        logger.debug("_get_project_object getting default subscription ID from auth because none was provided")
        subscription_id = get_default_subscription_id(auth)

    project_object = None
    try:
        # We don't want to check if workspace exits or project even exists in service in this case
        # As we are using Project object just as a wrapper around project scope.
        logger.debug("Attempt to instantiate a Project at %s", project_path)
        project_object = Project(auth=auth, directory=project_path,
                                 _disable_service_check=True)
    except Exception:
        pass

    if project_object:
        if project_object.workspace.subscription_id != subscription_id:
            raise UserErrorException("Default CLI subscription id = {}, which doesn't match "
                                     "with the project subscription id = {} "
                                     "at location = {}".format(subscription_id,
                                                               project_object.workspace.subscription_id,
                                                               project_object.project_directory))
    return project_object


def _parse_key_values(text, description):
    """
    Parse a comma-separated list of key=value pairs, and return as a dictionary.
    :param text: text to parse
    :param description:  Description of the items being parsed (to show in an error message)
    :return: dict
    """
    parsed_items = {}
    if text is not None:
        for key_value in text.split(','):
            if '=' not in key_value:
                raise UserErrorException("%s %s should be a key=value pair" % (description, key_value))
            tokens = key_value.split('=', 1)
            parsed_items[tokens[0]] = tokens[1]

    return parsed_items
