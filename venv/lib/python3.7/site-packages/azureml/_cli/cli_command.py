# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
from functools import wraps

from azureml._base_sdk_common.common import set_correlation_id
from azureml._cli import argument

from azureml.core import get_run
from azureml.exceptions import AzureMLException
from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import UnsupportedReturnType, FileAlreadyExists

module_logger = logging.getLogger(__name__)


class CliCommand(object):
    def __init__(self, name, title, arguments, handler_function_path, *, description=None, examples=None):
        self._name = name
        self._title = title
        self._description = description
        self._arguments = arguments
        self._examples = examples
        self._handler_function_path = handler_function_path

    def get_command_name(self):
        """Returns the name of the command. This name will be used in the cli command."""
        return self._name

    def get_command_title(self):
        """Returns the command title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the command."""
        return self._title

    def get_command_description(self):
        """Returns the command description as string. This describes the command in details
        and could be in multi-line."""
        return self._description

    def get_command_arguments(self):
        """An abstract method to return command arguments.
        The arguments are returned as a list of Argument class objects."""
        return self._arguments

    def get_examples(self):
        """Get example entries for the command.
        Refer to `azureml._cli.example.Example` for details."""
        return self._examples or []

    def get_handler_function_path(self):
        """Returns the string representation of the handler function for this amlcli command.
        The function name format is module_name#function_name. The module_name should be
        resolvable based on sys.path.
        An example is azure.cli.command_modules.machinelearning.cmd_experiment#start_run"""
        return self._handler_function_path

    def get_cli_to_function_arg_map(self):
        """
        Returns the cli to handler function argument name mapping.
        :return:
        """

        mapping_dict = {}
        if self._arguments is not None:
            for arg in self._arguments:
                if isinstance(arg.long_form, str):
                    long_form_str = arg.long_form
                elif isinstance(arg.long_form, list):
                    # Just taking first element from list.
                    long_form_str = arg.long_form[0]
                else:
                    assert False, "{} is not of type str or list".format(arg.long_form)

                if long_form_str.startswith("--"):
                    arg_name = long_form_str[2:]
                else:
                    arg_name = long_form_str

                mapping_dict[arg_name.replace("-", "_")] = arg.function_arg_name
        return mapping_dict


def _to_camel_case(snake_str):
    # TODO: Figure out a dedupe story for these 3 functions
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


def _process_single_return_object(return_object):
    from msrest.serialization import Model
    if isinstance(return_object, Model):
        object_dict = return_object.as_dict()
        return {_to_camel_case(k): v for k, v in object_dict.items()}
    elif isinstance(return_object, dict):
        return return_object
    else:
        azureml_error = AzureMLError.create(
            UnsupportedReturnType, return_object=type(return_object)
        )
        raise AzureMLException._with_error(azureml_error)


def _convert_return_to_json(return_object):
    from collections import Iterator
    if return_object:
        if isinstance(return_object, (list, Iterator)):
            return [_process_single_return_object(obj) for obj in return_object]
        else:
            return _process_single_return_object(return_object)


def _write_output_metadata_file(return_object, output_metadata_file_path, logger):
    import errno
    import json
    import os
    logger.debug(
        "Specified output metadata path %s, storing return value type [%s] as json",
        output_metadata_file_path,
        type(return_object).__name__)

    # TODO: Move this to a file utils library in azureml-core
    full_path = os.path.abspath(output_metadata_file_path)
    if os.path.exists(full_path):
        # Can't just use 'x' in open() mode below due to Python 2
        azureml_error = AzureMLError.create(
            FileAlreadyExists, full_path=full_path
        )
        raise AzureMLException._with_error(azureml_error)

    dir_path = os.path.dirname(full_path)
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as exc:
            # No Python 3 guarantee for exist_ok :(
            if exc.errno != errno.EEXIST:
                raise

    with open(output_metadata_file_path, 'wt') as omf:
        json.dump(_convert_return_to_json(return_object), omf, indent=4, sort_keys=True)


class AutoArg(object):
    Workspace = 'workspace'
    Experiment = 'experiment'
    Run = 'run'


auto_workspace_arguments = [
    # We can in the decorator auto-add these for workspace= if the signature has it
    argument.SUBSCRIPTION_ID,
    argument.RESOURCE_GROUP_NAME,
    argument.WORKSPACE_NAME,
    argument.PROJECT_PATH
]

auto_experiment_arguments = list(auto_workspace_arguments)
auto_experiment_arguments.append(argument.EXPERIMENT_NAME)

auto_run_arguments = list(auto_experiment_arguments)
auto_run_arguments.append(argument.RUN_ID_OPTION.get_required_true_copy())

auto_args_mapping = {
    AutoArg.Workspace: auto_workspace_arguments,
    AutoArg.Experiment: auto_experiment_arguments,
    AutoArg.Run: auto_run_arguments
}


def command(subgroup_type=None, command=None, short_description=None, argument_list=None,
            long_description=None, examples=None):
    # Require the args to be keyword only for clarity. But kwonly args without defaults aren't
    # supported in Python 2, so default to None and assert internally that they're set
    # TODO: We can probably even drop this on AbstractSubGroup and get even sweeter syntax/autoinfer
    assert subgroup_type is not None
    assert command is not None
    assert short_description is not None

    # Put this here to avoid sad recursive imports. Fix the arch later
    from azureml._base_sdk_common.cli_wrapper._common import get_workspace_or_default, _get_experiment_or_default

    if argument_list is None:
        argument_list = []

    def command_decorator(function):

        import inspect
        import six
        if six.PY3:
            sig = inspect.getfullargspec(function)
            func_arglist = sig.args + sig.kwonlyargs
        else:
            sig = inspect.getargspec(function)
            func_arglist = sig.args
        module_logger.debug("Decorating function [%s] with signature %s", function.__name__, sig)
        module_logger.debug("Deduced args %s", func_arglist)

        # Higher-level arguments to be auto-populated by the CLI infra
        command_auto_args = []
        # User-facing args that were injected by the above
        command_auto_args_only = []

        # Inject structured object output for all commands
        argument_list.append(argument.OUTPUT_METADATA_FILE)
        command_auto_args_only.append(argument.OUTPUT_METADATA_FILE.function_arg_name)

        for auto_arg in auto_args_mapping.keys():
            if auto_arg in func_arglist:
                command_auto_args.append(auto_arg)
                # I think we can just do set merge, but need to check why they weren't already used
                for arg in auto_args_mapping[auto_arg]:
                    if arg not in argument_list:
                        argument_list.append(arg)
                        # Keep track of params we need to pop off
                        command_auto_args_only.append(arg.function_arg_name)

        @wraps(function)
        def command_wrapper(*args, **kwargs):
            # Diagnostics
            command_logger = kwargs.get('logger')
            if command_logger is None:
                # The az cli framework will take control of the loggers that start with a `cli.` namespace.
                # The logger output level will be set according to the --debug, --verbose, --only-show-errors flags.
                #
                # On the other hand, for loggers that without a `cli.` namespace, the cli framework will
                # set to CRITICAL by default, regardless of --verbose, --only-show-errors flags.
                #
                # Related code could be referenced here:
                # https://github.com/microsoft/knack/blob/fe3bf5d3a79a3dd2ce5ddb0c38d93843a3380f6f/knack/log.py#L179
                #
                # Here we try to get the logger from the az cli framework first.
                # If failed, (e.g. When doing a local debug from `azml` command that may not have
                # the `knack` package installed), fallback to the default logger.
                try:
                    from knack.log import get_logger
                    command_logger = get_logger(function.__module__)
                except ImportError:
                    command_logger = logging.getLogger(function.__module__)

            kwargs['logger'] = command_logger
            command_logger.debug("Invoked {} with args {} and kwargs {}".format(function.__name__, args, kwargs))

            def _get(kwargs, arg, default=None):
                argname = arg.function_arg_name
                pop = argname in command_auto_args_only
                ret = kwargs.pop(argname, default) if pop else kwargs.get(argname, default)
                command_logger.debug("Popping auto argument %s: %s => %s", argname, pop, ret)
                return ret

            # Call cli-agnostic correlation setter
            set_correlation_id()

            # Auto load workspace if declared
            if AutoArg.Workspace in command_auto_args or AutoArg.Experiment in command_auto_args or \
               AutoArg.Run in command_auto_args:
                sub_id = _get(kwargs, argument.SUBSCRIPTION_ID)
                rg_name = _get(kwargs, argument.RESOURCE_GROUP_NAME)
                ws_name = _get(kwargs, argument.WORKSPACE_NAME)
                proj_path = _get(kwargs, argument.PROJECT_PATH, ".")
                command_logger.debug("Hydrating auto arg workspace from "
                                     "subscription %s, "
                                     "resource_group %s, "
                                     "workspace name %s, "
                                     "project path %s", sub_id, rg_name, ws_name, proj_path)
                workspace = get_workspace_or_default(
                    subscription_id=sub_id,
                    resource_group=rg_name,
                    workspace_name=ws_name,
                    auth=None,
                    project_path=proj_path,
                    logger=command_logger
                )

            if AutoArg.Experiment in command_auto_args or AutoArg.Run in command_auto_args:
                experiment_name = _get(kwargs, argument.EXPERIMENT_NAME)
                command_logger.debug("Hydrating auto arg experiment %s", experiment_name)
                experiment = _get_experiment_or_default(
                    workspace=workspace,
                    experiment_name=experiment_name,
                    project_path=proj_path,
                    logger=command_logger
                )

            if AutoArg.Run in command_auto_args:
                run_id = _get(kwargs, argument.RUN_ID_OPTION)
                command_logger.debug("Hydrating auto arg run %s", run_id)
                run = get_run(experiment, run_id)

            if AutoArg.Workspace in command_auto_args:
                kwargs[AutoArg.Workspace] = workspace
            if AutoArg.Experiment in command_auto_args:
                kwargs[AutoArg.Experiment] = experiment
            if AutoArg.Run in command_auto_args:
                kwargs[AutoArg.Run] = run

            # Handle common -t output for all commands
            output_metadata_file_path = _get(kwargs, argument.OUTPUT_METADATA_FILE)

            # Call the underlying command function
            command_logger.debug("Calling %s with args %s and kwargs %s", function.__name__, args, kwargs)
            retval = function(*args, **kwargs)

            if output_metadata_file_path:
                _write_output_metadata_file(retval, output_metadata_file_path, command_logger)

            return retval

        # Build the cli-agnostic Command object used for arg/command parsing
        function_string = "{}#{}".format(function.__module__, function.__name__)
        command_object = CliCommand(command, short_description, argument_list, function_string,
                                    description=long_description, examples=examples)

        subgroup_type.add_command(command_object)

        return command_wrapper

    return command_decorator
