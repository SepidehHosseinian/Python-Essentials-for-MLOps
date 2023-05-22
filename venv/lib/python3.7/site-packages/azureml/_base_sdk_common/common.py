# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" base_sdk_common.py, A file for storing commonly-used functions."""
from __future__ import print_function

import json
import logging
import os
import re
import sys
import uuid
import warnings
import jwt

from os import listdir
from os.path import isfile, join

from azureml.exceptions import ProjectSystemException, UserErrorException, AuthenticationException

module_logger = logging.getLogger(__name__)


TOKEN_EXPIRE_TIME = 5 * 60
AZ_CLI_AAP_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'

# EXTENSIONS AND FILE NAMES
RUNCONFIGURATION_EXTENSION = '.runconfig'
COMPUTECONTEXT_EXTENSION = '.compute'
YAML_EXTENSION = '.yml'
AML_CONFIG_DIR = "aml_config"
AZUREML_DIR = ".azureml"

LEGACY_PROJECT_FILENAME = "project.json"
CONFIG_FILENAME = "config.json"

TEAM_FILENAME = '.team'
ACCOUNT_FILENAME = 'runhistory'

# ARM RELATED CONSTANTS
ARM_ACCOUNT_DATA = "ARM_TEAM"
TEAM_LIST_OF_KEYS = {"subscriptions", "resourceGroups", "accounts", "workspaces"}
TEAM_DEFAULT_KEY = "id"
ACCOUNT_DEFAULT_KEY = "default"
CORRELATION_ID = None

# EXPERIMENT LENGTH CONSTRAINTS
EXPERIMENT_LENGTH_MAX = 255
EXPERIMENT_LENGTH_MIN = 1

# Environment variable names related to arm account token and user's email address.
# UX or any other service can set these environment variables in the python
# environment then the code uses values of these variables
AZUREML_ARM_ACCESS_TOKEN = "AZUREML_ARM_ACCESS_TOKEN"

# The subscription id environment variable. Mainly used for project commands.
AZUREML_SUBSCRIPTION_ID = "AZUREML_SUBSCRIPTION_ID"

# Environment variable for tenant id, mainly required for flighting.
AZUREML_TENANT_ID = "AZUREML_TENANT_ID"


# Default resource group location.
# Currently, supported locations are: australiaeast, eastus2, westcentralus, southeastasia, westeurope, eastus2euap
# TODO: Should be changed to eastus

# If we are running this code from source then we use default region as eastus2euap
if os.path.abspath(__file__).endswith(
        os.path.join("src", "azureml-core", "azureml", "_base_sdk_common", "common.py")):
    DEFAULT_RESOURCE_LOCATION = "eastus2euap"
else:
    DEFAULT_RESOURCE_LOCATION = "eastus2"


# FILE LOCATIONS
if 'win32' in sys.platform:
    USER_PATH = os.path.expanduser('~')
    # TODO Rename CREDENTIALS_PATH since there aren't credentials there anymore.
    CREDENTIALS_PATH = os.path.join(USER_PATH, ".azureml")
else:
    USER_PATH = os.path.join(os.getenv('HOME'), '.config')
    CREDENTIALS_PATH = os.path.join(os.getenv('HOME'), '.azureml')


def normalize_windows_paths(path):
    if not path:
        return path
    if os.name == "nt":
        return path.replace("\\", "/")

    return path


# PROJECT RELATED METHODS


def _valid_workspace_name(workspace_name):
    """Check validity of workspace name"""
    if not workspace_name:
        return False
    return re.match("^[a-zA-Z0-9][\w\-]{2,32}$", workspace_name)


def _valid_experiment_name(experiment_name):
    """Check validity of experiment name"""
    if not experiment_name:
        return False
    regex = "^[a-zA-Z0-9][\w\-]{{{},{}}}$".format(EXPERIMENT_LENGTH_MIN-1, EXPERIMENT_LENGTH_MAX-1)
    return re.match(regex, experiment_name)


def fetch_tenantid_from_aad_token(token):
    # We set verify=False, as we don't have keys to verify signature, and we also don't need to
    # verify signature, we just need the tenant id.
    decode_json = jwt.decode(token, options={'verify_signature': False, 'verify_aud': False})
    return decode_json['tid']


# only return run config path it if exists either in aml_config or .azureml
# folder. Else simply return project path
def get_run_config_dir_path_if_exists(project_path):
    # Try to look for the old aml_config directory first
    # If that does not exist default to use the new .azureml
    run_config_dir_path = os.path.join(project_path, AML_CONFIG_DIR)
    if not os.path.exists(run_config_dir_path):
        run_config_dir_path = os.path.join(project_path, AZUREML_DIR)
        if not os.path.exists(run_config_dir_path):
            return project_path
    return run_config_dir_path


def get_run_config_dir_path(project_path):
    # Try to look for the old aml_config directory first
    # If that does not exist default to use the new .azureml
    run_config_dir_path = os.path.join(project_path, AML_CONFIG_DIR)
    if not os.path.exists(run_config_dir_path):
        run_config_dir_path = os.path.join(project_path, AZUREML_DIR)
    return run_config_dir_path


def get_run_config_dir_name(project_path):
    # Try to look for the old aml_config directory first
    # If that does not exist default to use the new .azureml
    run_config_dir_path = os.path.join(project_path, AML_CONFIG_DIR)
    run_config_dir_name = AML_CONFIG_DIR
    if not os.path.exists(run_config_dir_path):
        run_config_dir_name = AZUREML_DIR
    return run_config_dir_name


def get_config_file_name(project_config_path):
    """
    :param project_config_path: Either project_path/aml_config or project_path/.azureml
    :type project_config_path: str
    :return: Either project.json or config.json
    :rtype: str
    """
    legacy_config_file_path = os.path.join(project_config_path, LEGACY_PROJECT_FILENAME)
    if os.path.exists(legacy_config_file_path):
        return LEGACY_PROJECT_FILENAME
    else:
        return CONFIG_FILENAME


def check_valid_resource_name(resource_name, resource_type):
    """
    Checks if the resource name is valid.
    If it is non valid then we throw UserErrorException.
    :param resource_name: The resource name.
    :type resource_name: str
    :param resource_type: The resource type like Workspace or History.
    :type resource_type: str
    :return:
    :rtype: None
    """
    message = "{} name must be between {} and {} characters long. " \
              "Its first character has to be alphanumeric, and " \
              "the rest may contain hyphens and underscores. " \
              "No whitespace is allowed."
    if resource_type.lower() == 'experiment':
        if not _valid_experiment_name(resource_name):
            raise UserErrorException(message.format(resource_type, EXPERIMENT_LENGTH_MIN, EXPERIMENT_LENGTH_MAX))
    elif not _valid_workspace_name(resource_name):
        raise UserErrorException(message.format(resource_type, 3, 33))


def graph_client_factory(auth):
    """
    :param auth:
    :type auth: azureml.core.authentication.AbstractAuthentication
    :return:
    """
    from azure.graphrbac import GraphRbacManagementClient
    tenant_id = fetch_tenantid_from_aad_token(auth._get_arm_token())
    client = GraphRbacManagementClient(auth._get_adal_auth_object(is_graph_auth=True),
                                       tenant_id,
                                       base_url=auth._get_cloud_type().endpoints.active_directory_graph_resource_id)
    return client


def auth_client_factory(auth, scope=None):
    """
    :param auth: auth object
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param scope:
    :return:
    """
    subscription_id = None
    if scope:
        subscription_id = _get_subscription_from_scope(scope)
    from azure.mgmt.authorization import AuthorizationManagementClient
    return auth._get_service_client(AuthorizationManagementClient, subscription_id=subscription_id)


def _get_subscription_from_scope(scope):
    matched = re.match('/subscriptions/(?P<subscription>[^/]*)/', scope)
    if matched:
        return matched.groupdict()['subscription']


# pylint:disable=too-many-arguments
def _resolve_role_id(role, scope, definitions_client):
    """
    # TODO: Types of input parameters.
    :param role:
    :param scope:
    :param definitions_client:
    :return:
    """
    role_id = None
    if re.match(r'/subscriptions/.+/providers/Microsoft.Authorization/roleDefinitions/',
                role, re.I):
        role_id = role
    else:
        try:
            # try to parse role as a guid, if fails then try to retrieve it
            uuid.UUID(role)
            role_id = '/subscriptions/{}/providers/Microsoft.Authorization/roleDefinitions/{}'.format(
                definitions_client.config.subscription_id, role)
        except ValueError:
            pass
        if not role_id:  # retrieve role id
            role_defs = list(definitions_client.list(scope, "roleName eq '{}'".format(role)))
            if not role_defs:
                raise ProjectSystemException("Role '{}' doesn't exist.".format(role))
            elif len(role_defs) > 1:
                ids = [r.id for r in role_defs]
                err = "More than one role matches the given name '{}'. Please pick a value from '{}'"
                raise ProjectSystemException(err.format(role, ids))
            role_id = role_defs[0].id
    return role_id


def _build_role_scope(resource_group_name, scope, subscription_id):
    subscription_scope = '/subscriptions/' + subscription_id
    if scope:
        if resource_group_name:
            err = 'Resource group "{}" is redundant because scope is supplied'
            raise ProjectSystemException(err.format(resource_group_name))
    elif resource_group_name:
        scope = subscription_scope + '/resourceGroups/' + resource_group_name
    else:
        scope = subscription_scope
    return scope


def _get_object_stubs(graph_client, assignees):
    from azure.graphrbac.models import GetObjectsParameters
    params = GetObjectsParameters(include_directory_object_references=True,
                                  object_ids=assignees)
    return list(graph_client.objects.get_objects_by_object_ids(params))


def _resolve_object_id(auth, assignee):
    """
    TODO: assignee type
    :param assignee:
    :return:
    """
    client = graph_client_factory(auth)
    result = None
    if assignee.find('@') >= 0:  # looks like a user principal name
        result = list(client.users.list(filter="userPrincipalName eq '{}'".format(assignee)))
    if not result:
        result = list(client.service_principals.list(
            filter="servicePrincipalNames/any(c:c eq '{}')".format(assignee)))
    if not result:  # assume an object id, let us verify it
        result = _get_object_stubs(client, [assignee])

    # 2+ matches should never happen, so we only check 'no match' here
    if not result:
        raise ProjectSystemException("No matches in graph database for '{}'".format(assignee))

    return result[0].object_id


def create_role_assignment(auth, role, assignee, resource_group_name=None, scope=None,
                           resolve_assignee=True):
    """
    :param auth: auth object
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param role:
    :param assignee:
    :param resource_group_name:
    :param scope:
    :param resolve_assignee:
    :return:
    """

    factory = auth_client_factory(auth, scope)
    assignments_client = factory.role_assignments
    definitions_client = factory.role_definitions

    scope = _build_role_scope(resource_group_name, scope,
                              assignments_client.config.subscription_id)

    role_id = _resolve_role_id(role, scope, definitions_client)
    object_id = _resolve_object_id(auth, assignee) if resolve_assignee else assignee

    from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
    properties = RoleAssignmentCreateParameters(role_definition_id=role_id, principal_id=object_id)
    assignment_name = uuid.uuid4()
    custom_headers = None
    return assignments_client.create(scope, assignment_name, properties, custom_headers=custom_headers)


def resource_error_handling(response_exception, resource_type):
    """General error handling for projects"""
    if response_exception.response.status_code == 404:
        raise ProjectSystemException("{resource_type} not found.".format(resource_type=resource_type))
    else:
        response_message = get_http_exception_response_string(response_exception.response)
        raise ProjectSystemException(response_message)


def resource_client_factory(auth, subscription_id):
    """
    Returns the azure SDK resource management client.
    :param auth: auth object
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :return:
    :rtype: azure.mgmt.resource.resources.ResourceManagementClient
    """
    from azure.mgmt.resource.resources import ResourceManagementClient
    return auth._get_service_client(ResourceManagementClient, subscription_id)


def storage_client_factory(auth, subscription_id):
    """
    Returns the azure SDK storage management client.
    :param auth: auth object
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :return:
    :rtype: azure.mgmt.resource.resources.StorageManagementClient
    """
    from azure.mgmt.storage import StorageManagementClient
    return auth._get_service_client(StorageManagementClient, subscription_id)


def get_project_id(project):
    """Gets project id from metadata"""
    project = os.path.join(os.path.dirname(os.getcwd()), '.ci')

    with open(os.path.join(project, 'metadata'), 'r') as file:
        metadata = json.load(file)

    if 'Id' in metadata:
        return metadata['Id']
    else:
        raise ValueError('No project id found.')

# COMMAND RELATED METHODS


def set_config_dir():
    """Get home directory"""
    if not os.path.exists(CREDENTIALS_PATH):
        os.makedirs(CREDENTIALS_PATH)

# ARM RELATED METHODS


def check_for_keys(key_list, dictionary):
    """Checks if all keys are present in dictionary"""
    return True if all(k in dictionary for k in key_list) else False


# CLI FUNCTIONALITY RELATED METHODS


def set_correlation_id():
    """Set telemetry correlation data with application information and newly-created correlation id"""
    # This function is called from CLI command handler functions that's why it can have azure.cli.core dependency
    try:
        from azure.cli.core.telemetry import set_module_correlation_data
    except ImportError:
        pass
    else:
        global CORRELATION_ID
        CORRELATION_ID = uuid.uuid4()
        module_logger.debug("Session correlation ID set to %s", CORRELATION_ID)
        set_module_correlation_data(str(CORRELATION_ID) + ' user')


def to_snake_case(input_string):
    """
    Converts a string into a snake case.
    :param input_string: Input string
    :return: Snake case string
    :rtype: str
    """
    final = ''
    for item in input_string:
        if item.isupper():
            final += "_"+item.lower()
        else:
            final += item
    if final[0] == "_":
        final = final[1:]
    return final


def to_title_case(input_string):
    camel_case = to_camel_case(input_string)
    title_case = camel_case[0].upper() + camel_case[1:]
    return title_case


def to_camel_case(snake_str):
    """
    Converts snake case and title case to camelCase.
    :param snake_str:
    :return:
    """
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    # We also just explicitly change the first character to lowercase, in case it is not, especially
    # in title case.
    return components[0][0].lower() + components[0][1:] + "".join(x.title() for x in components[1:])


def convert_dict_keys_to_camel_case(input_dict):
    return {to_camel_case(k): v for k, v in input_dict.items()}


def get_project_files(name, suffix):
    """
    Returns all project config files with the specified suffix.
    :param name:
    :type name: str
    :param suffix:
    :type suffix: str
    :return: Project files with specified suffix
    :rtype: dict
    """
    if suffix in name:
        # Prevent error for users who type the filename with the suffix included
        name = name[:-len(suffix)]
    targets = []
    run_config_path = get_run_config_dir_path_if_exists(name)
    files = [f for f in listdir(run_config_path) if isfile(join(run_config_path, f))]
    for file in files:
        if file.endswith(suffix):
            targets.append(os.path.relpath(os.path.join(run_config_path, file), name))
    return {os.path.splitext(os.path.basename(x))[0]: x for x in targets}


def give_warning(warning_message):
    # A custom formatter, so that we don't print the ugly full file path
    # in the userfacing warning.

    def custom_formatwarning(message, category, filename, lineno, line=None):
        # Ignore everything except the message
        return str(message) + '\n'

    original_formatter = warnings.formatwarning
    warnings.formatwarning = custom_formatwarning
    warnings.warn(warning_message, UserWarning)
    warnings.formatwarning = original_formatter


def get_http_exception_response_string(http_response):
    """
    Takes a http_response and returns a json formatted string with
    appropriate fields.
    :param http_response:
    :type http_response: requests.Response
    :return:
    :rtype: str
    """
    error_dict = dict()
    error_dict["url"] = http_response.url
    error_dict["status_code"] = http_response.status_code
    if http_response.text:
        try:
            # In case response is a json, which is the usual case with service responses.
            error_dict["error_details"] = remove_empty_values_json(http_response.json())
        except BaseException:
            # Response is not a json, so we just add that as text.
            error_dict["error_details"] = http_response.text
    return json.dumps(error_dict, indent=4, sort_keys=True)


def remove_empty_values_json(data):
    try:
        if not isinstance(data, (dict, list)):
            return data
        if isinstance(data, list):
            return [v for v in (remove_empty_values_json(v) for v in data) if v]
        return {k: v for k, v in ((k, remove_empty_values_json(v)) for k, v in data.items()) if v}
    except BaseException:
        return data


def perform_interactive_login(username=None, password=None, service_principal=None, tenant=None,
                              allow_no_subscriptions=False, identity=False, use_device_code=False,
                              use_cert_sn_issuer=None, cloud_type=None):
    """Log in to access Azure subscriptions"""
    from azureml._vendor.azure_cli_core._session import ACCOUNT, CONFIG, SESSION
    from azureml._vendor.azure_cli_core._environment import get_config_dir
    from azureml._vendor.azure_cli_core.util import in_cloud_console
    from azureml.core.authentication import _get_profile
    
    ACCOUNT.load(os.path.join(get_config_dir(), 'azureProfile.json'))
    CONFIG.load(os.path.join(get_config_dir(), 'az.json'))
    SESSION.load(os.path.join(get_config_dir(), 'az.sess'),
                 max_age=3600)

    _CLOUD_CONSOLE_LOGIN_WARNING = (
        "Cloud Shell is automatically authenticated under the initial account signed-in with."
        " Run 'az login' only if you need to use a different account")

    # quick argument usage check
    if any([password, service_principal, tenant, allow_no_subscriptions]) and identity:
        raise AuthenticationException("usage error: '--identity' is not applicable with other arguments")
    if any([password, service_principal, username, identity]) and use_device_code:
        raise AuthenticationException("usage error: '--use-device-code' is not applicable with other arguments")
    if use_cert_sn_issuer and not service_principal:
        raise AuthenticationException("usage error: '--use-sn-issuer' is only applicable with a service principal")
    if service_principal and not username:
        raise AuthenticationException('usage error: --service-principal --username NAME --password SECRET '
                                      '--tenant TENANT')

    interactive = False

    profile = _get_profile(cloud_type)

    if identity:
        if in_cloud_console():
            return profile.login_in_cloud_shell()
        return profile.login_with_managed_identity(identity_id=username)
    elif in_cloud_console():  # tell users they might not need login
        import logging
        logging.getLogger(__name__).warning(_CLOUD_CONSOLE_LOGIN_WARNING)

    if username:
        if not password:
            raise UserErrorException('Please specify both username and password in non-interactive mode.')
    else:
        interactive = True

    import requests
    from msal.exceptions import MsalError
    from azureml._vendor.azure_cli_core.auth.identity import ServicePrincipalAuth
    credential = ServicePrincipalAuth.build_credential(secret_or_certificate=password)
    try:
        subscriptions = profile.login(
            interactive,
            username,
            credential,
            service_principal,
            tenant,
            use_device_code=use_device_code,
            allow_no_subscriptions=allow_no_subscriptions,
            use_cert_sn_issuer=use_cert_sn_issuer)
    except MsalError as err:
        # try polish unfriendly server errors
        if username:
            msg = str(err)
            suggestion = "For cross-check, try 'az login' to authenticate through browser."
            if ('ID3242:' in msg) or ('Server returned an unknown AccountType' in msg):
                raise AuthenticationException("The user name might be invalid. " + suggestion)
            if 'Server returned error in RSTR - ErrorCode' in msg:
                raise AuthenticationException("Logging in through command line is not supported. " + suggestion)
        raise AuthenticationException('Unknown error occurred during authentication. Error detail: ' + str(err))
    except requests.exceptions.ConnectionError as err:
        raise AuthenticationException('Please ensure you have network connection. Error detail: ' + str(err))
    all_subscriptions = list(subscriptions)
    for sub in all_subscriptions:
        sub['cloudName'] = sub.pop('environmentName', None)
    return all_subscriptions


class CLICommandError(object):
    # TODO: This class needs to be removed or merged with our new excepion classes
    """
    A class for returning a error for a CLI command. This class is mainly used
    when an error is not because of a service request. In the case of service errors, we
    directly return the error json returned by a service.
    """

    def __init__(self, error_type, error_message, stack_trace=None, **kwargs):
        """
        CLICommandError constructor.
        :param error_type: Type of error like UserError, FormattingError etc.
        :param error_message: Error message.
        :param stack_trace: Stack trace if available.
        :param kwargs: A dictionary of any other key-value pairs to include in the error message.
        :type error_type: str
        :type error_message: str
        :type stack_trace: str
        :type kwargs: dict
        """
        self._error_type = error_type
        self._error_message = error_message
        self._stack_trace = stack_trace
        self._kwargs = kwargs

    def get_json_dict(self):
        """
        Serializes the object into a dictionary, which can be printed as a JSON object.
        :return: A dictionary representation of this object.
        :rtype: dict
        """
        error_dict = {"errorType": self._error_type, "errorMessage": self._error_message}

        if self._stack_trace:
            error_dict["stackTrace"] = self._stack_trace

        if self._kwargs:
            for key, value in self._kwargs.items():
                error_dict[key] = value

        return error_dict


class CLICommandOutput(object):
    # TODO: This class also needs to be removed, and sdk methods should return specific
    # return types.
    """
    A class for returning a command output for a CLI command.
    """

    def __init__(self, command_output):
        """
        CLICommandOutput constructor.
        :param command_output: Output of a command. command_output can contain
        multiple lines separated by \n
        :type command_output: str
        """
        self._command_output = command_output
        self._dict_to_merge = None
        self._do_not_print_dict = False

    def get_json_dict(self, exclude_command_output=False):
        """
        Serializes the object into a dictionary, which can be printed as a JSON object.
        :param exclude_command_output: exclude_command_output=True, excludes the commandOutput key from
        the returned dictionary.
        :return: A dictionary representation of this object.
        :rtype: dict
        """
        if self._dict_to_merge:
            if not exclude_command_output:
                self._dict_to_merge["commandOutput"] = self._command_output

            return self._dict_to_merge
        else:
            if exclude_command_output:
                return {}
            else:
                output_dict = {"commandOutput": self._command_output}
                return output_dict

    def append_to_command_output(self, lines_to_append):
        """
        Appends lines to the command output.
        Basically, this function does _command_output=_command_output+"\n"+lines_to_append
        :param lines_to_append:
        :return: None
        """
        if len(self._command_output) > 0:
            self._command_output = self._command_output + "\n" + lines_to_append
        else:
            self._command_output = lines_to_append

    def merge_dict(self, dict_to_merge):
        """
        Merges dict_to_merge with the overall dict returned by get_json_dict.
        Basically, we merge the dict returned by a service with the CLI command output.
        The actual merge happens when get_json_dict is called.
        :param dict_to_merge:
        :return:
        """
        import copy
        self._dict_to_merge = copy.deepcopy(dict_to_merge)

    def get_command_output(self):
        """
        :return: Returns the command output.
        :rtype: str
        """
        return self._command_output

    def set_do_not_print_dict(self):
        """
        Set the flag to not print dict. Sometimes, in az CLI commands, we set the output in
        self._command_output in pretty format, and don't want to print the whole dict.
        :return:
        """
        self._do_not_print_dict = True

    def get_do_not_print_dict(self):
        """
        Returns True if az CLI should not print the output in dict/JSONObject format.
        :return:
        :rtype bool:
        """
        return self._do_not_print_dict
