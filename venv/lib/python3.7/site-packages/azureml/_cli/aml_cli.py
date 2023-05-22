# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import collections
import importlib
import json
import logging
import pkg_resources
import sys
import textwrap
import inspect
from pathlib import Path

from azureml._cli import abstract_subgroup

# A named tuple to store a parent parser, its add_subparser_handle and a subgroup while performing
# the breadth-first search over packages to register all commands.
# add_subparser_handle is obtained by calling parser.add_subparsers(), and for a parser add_subparsers()
# function cannot be called multiple times, so we need to call it once and maintain a handle to it.
# subgroup is an object of a derived class of AbstractSubGroup.
SubgroupParserInfo = collections.namedtuple("SubgroupParserInfo", "parent_parser add_subparser_handle subgroup")

# Named tuple that stores a subgroup and the the partial command name till the subgroup.
SubgroupCommandNameInfo = collections.namedtuple("SubgroupCommandNameInfo", "subgroup partial_command_name")


# Environment variable to indicate that the azureml._cli is in use.
# Mainly used for output formatting. azureml._cli returns JSON formatted outputs while
# az CLI can return non JSON outputs for pretty printing for users. Also used for some
# options that are only present in azureml._cli and not in az ml CLI.

# This env variable is here and not in azureml._base_sdk_common.base_sdk_common.py because
# azureml._base_sdk_common.base_sdk_common.py has high importing latency, which will slow down the azureml._cli
AZUREML_CLI_IN_USE = None

CLI_SUBGROUP_PROVIDERS_ENTRYPOINT_KEY = "azureml_cli_subgroup_providers"

module_logger = logging.getLogger(__name__)


class AmlCli(abstract_subgroup.AbstractSubGroup):

    _subgroup_plugins = {}

    @staticmethod
    def add_subgroup_providers_from_entry_points():
        """Dynamic loading custom cli subgroup from entry_points config.

        Example config in setup.py:
        entry_points={ 'azureml_cli_subgroup_providers': [
             'example = azureml.package.example_subgroup:ExampleSubGroup',
        ]},
        """

        # Skip loading plugin subgroups when plugin dict is not empty
        if len(AmlCli._subgroup_plugins) > 0:
            return
        # the default global working set used by pkg_resources won't refresh automatically.
        # so we have to reload it to make sure changes in sys.path synced.
        importlib.reload(pkg_resources)
        entry_points = [e for e in pkg_resources.iter_entry_points(CLI_SUBGROUP_PROVIDERS_ENTRYPOINT_KEY)]
        # the distribution folder for cli extension will be `xxx\.azure\cliextensions\azure-cli-ml\`
        # while default distribution folder for python is `xxx\Lib\site-packages\`
        # if package with entry points only installed for extension, pkg_resources couldn't load it.
        # in that case, we try to load entry point from cli extension's distribution folder.
        if len(entry_points) == 0:
            # a little bit hack here, when installed inside extension azure-cli-ml
            # this file path will be located in path `xxx/azure-cli-ml/azureml/_cli/aml_cli.py`
            # we need to go to folder `xxx/azure-cli-ml` to load distributions.
            base_dir = Path(__file__).parent.parent.parent.absolute().resolve()
            AmlCli.add_subgroup_providers_from_distributions(base_dir)
        else:
            for entrypoint in entry_points:
                AmlCli.add_subgroup_provider_from_entry_point(entrypoint)

    @staticmethod
    def add_subgroup_providers_from_distributions(distribution_base_dir):
        """Add custom cli subgroup from cli extension distributions.
        1. pkg_resources find all distributions from distribution_base_dir
        2. load entry point for each distribution
        3. import subgroup's package and class form entry point

        :param distribution_base_dir: base dir for distributions.
        """
        try:
            dists = pkg_resources.find_distributions(distribution_base_dir)
            # only load azure-ml or azureml distributions
            dists = filter(lambda dist: dist.key.startswith('azure-ml') or dist.key.startswith('azureml'), dists)
            for dist in dists:
                # only distributions with entry point key CLI_SUBGROUP_PROVIDERS_ENTRYPOINT_KEY will be loaded
                entry_point_dict = dist.get_entry_map(CLI_SUBGROUP_PROVIDERS_ENTRYPOINT_KEY)
                for name, entrypoint in entry_point_dict.items():
                    AmlCli.add_subgroup_provider_from_entry_point(entrypoint)
        except BaseException as e:
            module_logger.warning(
                "Failure while loading {} from cli extension's folder with exception {}.".format(
                    CLI_SUBGROUP_PROVIDERS_ENTRYPOINT_KEY, e)
            )

    @staticmethod
    def add_subgroup_provider_from_entry_point(entrypoint):
        """Add custom cli subgroup from entry point.

        :param entrypoint: entrypoint.
        :type entrypoint: pkg_resources.EntryPoint
        """
        try:
            # Use the following code to import plugin subgroup's whole package
            # because @command decorated command is only registered when imported.
            # Extract subpackage from entry point, eg: for entry point
            # azureml.package.example_subgroup:ExampleSubGroup, command sub package will be azureml.package
            subpackage = entrypoint.module_name.rsplit('.', 1)[0]
            for module in abstract_subgroup.AbstractSubGroup.get_all_modules(subpackage):
                importlib.import_module(subpackage + "." + module)

            AmlCli.load_subgroup_class_from_entry_point(entrypoint)
        except Exception as e:
            module_logger.warning("Failure while loading {}. Failed to load entrypoint {} with exception {}.".format(
                CLI_SUBGROUP_PROVIDERS_ENTRYPOINT_KEY, entrypoint, e))

    @staticmethod
    def load_subgroup_class_from_entry_point(entry_point):
        """Load custom cli subgroup's class from entry point.

        :param entrypoint: entry point.
        :return: entrypoint: pkg_resources.EntryPoint
        """
        # We just import class from entry point instead of calling entrypoint.load().
        # Because entrypoint.load() will try to parse the entry point's dependencies
        # and it might fail with error message "Requirement.parse('some-package'), {'some-package'}"

        # load subgroup class
        module_object = importlib.import_module(entry_point.module_name)
        name_class_tuples = inspect.getmembers(module_object, inspect.isclass)
        name_class_dicts = dict(name_class_tuples)
        if len(entry_point.attrs) != 1:
            raise ValueError('Entry point {} should have 1 class name configured, got {}'.format(
                entry_point.module_name, entry_point.attrs))
        class_name = entry_point.attrs[0]

        if class_name not in name_class_dicts.keys():
            raise ValueError('Failed to get class {} from module {}, got {}'.format(
                class_name, entry_point.module_name, name_class_dicts
            ))
        entry_class = dict(name_class_tuples)[class_name]

        # add provider
        AmlCli.add_subgroup_provider(entry_point.name, entry_class)

    @staticmethod
    def add_subgroup_provider(name, subgroup_class):
        """Extensibility hook for custom cli subgroup.

        :param name: name of the subgroup.
        :type name: str
        :param subgroup_class: A class inherits from AbstractSubGroup.
        :type subgroup_class: function
        """
        if not issubclass(subgroup_class, abstract_subgroup.AbstractSubGroup):
            raise ValueError("EntryPoint {0}: {1} is not a subgroup".format(name, subgroup_class))

        if AmlCli._subgroup_plugins.get(name, None) is not None:
            raise ValueError("Subgroup {0} already added, can't add another".format(name))

        assert abstract_subgroup.AbstractSubGroup._check_for_empty_constructor(subgroup_class), \
            "class={0} doesn't have empty constructor, " \
            "which is not supported for aml_cli.".format(subgroup_class)

        class_object = subgroup_class()
        AmlCli._subgroup_plugins[name] = class_object

    def get_subgroup_name(self):
        """Returns the name of the subgroup.
        This name will be used in the cli command."""
        return "azureml._cli"

    def get_subgroup_title(self):
        """Returns the subgroup title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the subgroup."""
        return "Manage Azure Machine Learning resources with the Azure CLI ML extension v1."

    def get_subgroup_description(self):
        """Returns the subgroup description as string. This describes the command group in details
        and could be in multi-line.."""
        long_summary = "[Warning] Azure ML CLI v1 is getting retired on September 30, 2025. " \
                       "Users are recommended to adopt CLI v2. Use documentation at https://aka.ms/migrateamlv2 " \
                       "and https://aka.ms/amlcliv2 to learn more about CLI v2." \
                       "\nInstall Azure CLI ML extension v1" \
                       "\nhttps://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli"
        return long_summary

    def get_nested_subgroups(self):
        """Returns sub-groups of this sub-group."""
        builtin_subgroups = super(AmlCli, self).compute_nested_subgroups(__package__)
        # Registers all plugin command subgroups.
        AmlCli.add_subgroup_providers_from_entry_points()
        plugin_subgroups = list(AmlCli._subgroup_plugins.values())
        all_subgroups = builtin_subgroups + plugin_subgroups
        all_subgroups.sort(key=lambda s: s.__class__.__name__)
        return all_subgroups

    def get_commands(self, for_azure_cli=False):
        """ Returns commands associated at this sub-group level."""
        from azureml._cli.cli_command import CliCommand
        return [] if for_azure_cli else [
            CliCommand(
                "login",
                "Interactive Login",
                [],
                "azureml._base_sdk_common.common#perform_interactive_login")
        ]


def _add_arguments_to_command_subparser(command_subparser, argument_list):
    """Adds arguments related to a command to its subparser.
    command_subparser is the subparser for the command.
    argument_list is a list of objects of Argument class."""

    if argument_list is None:
        return

    for argument_obj in argument_list:
        # Only adding those arguments that are not None, so that the unspecified arguments take their subparser default
        # values, and we don't have to deal with default values.

        # Empty positional arguments give exceptions, so we have to check if short form or long form argument is ""
        positional_argument_list = []
        if len(argument_obj.short_form) > 0:
            positional_argument_list.append(argument_obj.short_form)
        if len(argument_obj.long_form) > 0:
            positional_argument_list = positional_argument_list + argument_obj.long_form

        command_subparser.add_argument(*positional_argument_list, **argument_obj.get_key_based_arguments_as_dict())


def _get_command_function_object(function_path):
    """ Returns a function object based on the supplied function name. The function
    name is assumed to be of module_name#funciton_name format, where module_name should
    be resolvable from sys.path."""
    assert len(function_path) > 0, "Invalid function name supplied={0}".format(function_path)

    split_list = function_path.split("#")
    assert len(split_list) == 2, "Invalid function name supplied={0}".format(function_path)
    module_object = importlib.import_module(split_list[0])
    function_object = getattr(module_object, split_list[1])
    return function_object


def _register_subgroups_and_commands(root_parser):
    """Performs a breadth-first search over the subgroups package structure and registers corresponding subparsers
    and commands."""
    aml_cli = AmlCli()
    subgroups_list = [SubgroupParserInfo(root_parser, root_parser.add_subparsers(
        help=aml_cli.get_subgroup_name() + " sub-commands help"), aml_cli)]

    def _wrap_text(text, width=80):
        """Given a text, wrap possible long lines.

        >>> long_text = "A long sentence which contains multiple words that cannot be displayed in one line"
        >>> print(_wrap_text(long_text, width=45))
        A long sentence which contains multiple words
        that cannot be displayed in one line

        >>> print(_wrap_text(long_text, width=90))
        A long sentence which contains multiple words that cannot be displayed in one line
        """
        if not text:
            return None

        return '\n'.join(textwrap.wrap(text, width=width, break_on_hyphens=False))

    def _indent(text, indent=4):
        """Given a text (could be multi-lined), add indent for each line by `indent` spaces.

        >>> _indent('hello')
        '    hello'

        >>> _indent('hello', indent=2)
        '  hello'
        """
        if not text:
            return None

        return textwrap.indent(text, prefix=' ' * indent)

    def _format_one_example(example):
        return '\n'.join((
            _wrap_text(example.name),
            _indent(_wrap_text(example.text)),
        ))

    def _format_command_examples(command_obj):
        if not command_obj or not command_obj.get_examples():
            return None

        examples = '\n\n'.join(_format_one_example(e)
                               for e in command_obj.get_examples())
        return '\n'.join((
            'Examples:',
            _indent(examples, indent=2),
        ))

    while len(subgroups_list) > 0:
        temp_subgroup_list = []
        for subgroup_parser_object in subgroups_list:

            # Registering subparsers for the commands first.
            commands_at_current_subgroup = subgroup_parser_object.subgroup.get_commands()
            for command_obj in commands_at_current_subgroup:
                command_subparser = subgroup_parser_object.add_subparser_handle.add_parser(
                    command_obj.get_command_name(),
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    help=command_obj.get_command_title(),
                    description=_wrap_text(command_obj.get_command_description()),
                    epilog=_format_command_examples(command_obj))
                _add_arguments_to_command_subparser(command_subparser, command_obj.get_command_arguments())

                # Setting the command handler function_path here.
                # Only putting the function path here, we are not importing the module and getting
                # a function object here because that is a slow process, as we only want to
                # do that on a command execution and not when a user is just browsing packages using -h option,
                # which will be fast if we delay loading modules on an actual command.
                command_subparser.set_defaults(function_path=command_obj.get_handler_function_path())

                command_subparser.set_defaults(cli_to_function_args=command_obj.get_cli_to_function_arg_map())
                # We don't need to add a subparser to command_subparser, as it is a command.

            # Registering subparsers for the subparsers at this level now.
            subgroups_at_current_subgroup = subgroup_parser_object.subgroup.get_nested_subgroups()
            for subgroup_obj in subgroups_at_current_subgroup:
                subgroup_subparser = subgroup_parser_object.add_subparser_handle.add_parser(
                    subgroup_obj.get_subgroup_name(), help=subgroup_obj.get_subgroup_title(),
                    description=_wrap_text(subgroup_obj.get_subgroup_description()))

                temp_subgroup_list.append(SubgroupParserInfo(subgroup_subparser, subgroup_subparser.add_subparsers(
                    help=subgroup_obj.get_subgroup_name() + " sub-command help"), subgroup_obj))
        subgroups_list = temp_subgroup_list


# A dictionary that stores <full_command_name, CliCommand> tuple.
# This dictionary is populated in load_command_table_azurecli
# and is used to register arguments for a command in load_arguments
_azure_command_dict = {}


def load_azurecli_help(helps):
    """
    Performs a bread-first search over the azureml._cli package structure and loads all the commands
    in az_command_loader's command table.

    This function doesn't register commands' arguments.
    Those are registered using the load_arguments function when an actual command is invoked from the
    command line.

    :param az_command_loader:
    :type az_command_loader: MachineLearningCommandsLoader, which extends from AzCommandsLoader
    :param _: We don't use this argument.
    :return: Returns none, as the function directly modifies az_command_loader's command table.
    """

    # Performing azure cli related imports.
    # Note: These import are time consuming, so we are performing them here instead of putting
    # them on the top of this file and slowing down azureml._cli

    top_level_command_name = "ml"
    aml_cli = AmlCli()
    subgroups_list = [SubgroupCommandNameInfo(aml_cli, top_level_command_name)]
    while len(subgroups_list) > 0:
        temp_subgroup_list = []
        for subgroup_object in subgroups_list:
            # We don't register flighting commands with az CLI.
            if not subgroup_object.subgroup.register_with_azure_cli():
                continue

            # Register the help for the subgroup
            help_dct = {
                'type': 'group',
                'short-summary': subgroup_object.subgroup.get_subgroup_title(),
                'long-summary': subgroup_object.subgroup.get_subgroup_description(),
            }
            helps[subgroup_object.partial_command_name] = json.dumps(help_dct)

            # Register the help for each command in the subgroup
            for command in subgroup_object.subgroup.get_commands(for_azure_cli=True):
                command_help_dct = {
                    'type': 'command',
                    'short-summary': command.get_command_title(),
                    'long-summary': command.get_command_description(),
                    'examples': [e._asdict() for e in command.get_examples()],
                }
                command_entry_name = subgroup_object.partial_command_name + ' ' + command.get_command_name()
                helps[command_entry_name] = json.dumps(command_help_dct)

            # Checking the subgroups for the current subgroup.
            for subgroup_obj in subgroup_object.subgroup.get_nested_subgroups():
                temp_subgroup_list.append(SubgroupCommandNameInfo(
                    subgroup_obj, subgroup_object.partial_command_name + " " + subgroup_obj.get_subgroup_name()))
        subgroups_list = temp_subgroup_list


def load_command_table_azurecli(az_command_loader, _):
    """
    Performs a bread-first search over the azureml._cli package structure and loads all the commands
    in az_command_loader's command table.

    This function doesn't register commands' arguments.
    Those are registered using the load_arguments function when an actual command is invoked from the
    command line.

    :param az_command_loader:
    :type az_command_loader: MachineLearningCommandsLoader, which extends from AzCommandsLoader
    :param _: We don't use this argument.
    :return: Returns none, as the function directly modifies az_command_loader's command table.
    """

    # Performing azure cli related imports.
    # Note: These import are time consuming, so we are performing them here instead of putting
    # them on the top of this file and slowing down azureml._cli

    from azure.cli.core.commands import CliCommandType
    # Exception handler behavior from our sdk
    from azureml._base_sdk_common.cli_wrapper._common import cli_exception_handler

    top_level_command_name = "ml"
    aml_cli = AmlCli()
    subgroups_list = [SubgroupCommandNameInfo(aml_cli, top_level_command_name)]

    while len(subgroups_list) > 0:
        temp_subgroup_list = []
        for subgroup_object in subgroups_list:
            # We don't register flighting commands with az CLI.
            if not subgroup_object.subgroup.register_with_azure_cli():
                continue

            # Generating immediate commands at this subgroup first.
            commands_at_current_subgroup = subgroup_object.subgroup.get_commands(for_azure_cli=True)

            if len(commands_at_current_subgroup) > 0:
                parsed = commands_at_current_subgroup[0].get_handler_function_path().split("#")
                assert len(parsed) == 2, "Wrong function handler path format: {}".format(
                    commands_at_current_subgroup[0].get_handler_function_path())

                function_handler_module = parsed[0]

                # This is basically a command group in azure CLI.
                # TODO : This is a place to add our table transformer if needed.
                cli_command_type = CliCommandType(operations_tmpl=function_handler_module + "#{}")

                with az_command_loader.command_group(subgroup_object.partial_command_name, cli_command_type) as g:
                    for command_obj in commands_at_current_subgroup:
                        curr_parsed = command_obj.get_handler_function_path().split("#")
                        assert len(parsed) == 2, "Wrong function handler path format: {}".format(
                            command_obj.get_handler_function_path())

                        assert curr_parsed[0] == function_handler_module, \
                            "Handler functions for all CLI commands in subgroup = {} should be in " \
                            "module = {}. Found = {}".format(
                                subgroup_object.partial_command_name,
                                function_handler_module,
                                curr_parsed[0])

                        g.command(command_obj.get_command_name(), curr_parsed[1],
                                  exception_handler=cli_exception_handler)
                        full_command_name = subgroup_object.partial_command_name + " " + command_obj.get_command_name()

                        # This is needed to disable function signature scraping by az cli
                        az_command_loader.command_table[full_command_name].arguments_loader = lambda: []
                        # We override the command description with our description.
                        az_command_loader.command_table[full_command_name].description = command_obj.get_command_title

                        # Adding an entry in the dictionary that will be used to register this command's arguments.
                        full_command_name = subgroup_object.partial_command_name + " " + command_obj.get_command_name()
                        _azure_command_dict[full_command_name] = command_obj

            # Checking the subgroups for the current subgroup.
            subgroups_at_current_subgroup = subgroup_object.subgroup.get_nested_subgroups()
            for subgroup_obj in subgroups_at_current_subgroup:
                temp_subgroup_list.append(SubgroupCommandNameInfo(
                    subgroup_obj, subgroup_object.partial_command_name + " " + subgroup_obj.get_subgroup_name()))
        subgroups_list = temp_subgroup_list


def load_arguments(az_command_loader, current_command):
    """
    This function loads arguments for current_command using az_command_loader.

    This function only loads arguments for current_command.
    Unlike az CLI codebase, this function doesn't load arguments of all commands,
    which improves the performance.

    :param az_command_loader:
    :type az_command_loader: MachineLearningCommandsLoader, which extends from AzCommandsLoader
    :param current_command: The current command name.
    :type current_command: str
    :return: Returns none, as the function directly modifies az_command_loader's command table.
    """

    if current_command not in _azure_command_dict:
        # This is not a azureml._cli command, it is some other command, so this function just
        # returns.
        return

    # TODO: Azure CLI also has a lot of fancy features that we are not using now,
    # but might want to use those some time.
    with az_command_loader.argument_context(current_command) as c:
        # Now registering each argument.
        for argument_obj in _azure_command_dict[current_command].get_command_arguments():
            # Only adding those arguments that are not None, so that the unspecified arguments
            # take their subparser default values, and we don't have to deal with default values.

            # Empty positional arguments give exceptions, so we have to check if short form or long
            # form argument is ""
            positional_argument_list = []
            if len(argument_obj.short_form) > 0:
                positional_argument_list.append(argument_obj.short_form)

            if len(argument_obj.long_form) > 0:
                positional_argument_list = positional_argument_list + argument_obj.long_form

            argument_name = argument_obj.function_arg_name

            arguments_dict = argument_obj.get_key_based_arguments_as_dict()

            # Adding "options_list" key, which is required by Azure CLI
            arguments_dict["options_list"] = positional_argument_list
            # Use extra instead of argument in conjunction with the empty
            # lambda to get the arguments flowing through
            c.extra(argument_name, **arguments_dict)

            # Adding these also to add back the "metadata" that we didn't register by overriding the lambda.
            if argument_obj.positional_argument:
                c.positional(argument_name, **arguments_dict)
            else:
                c.argument(argument_name, **arguments_dict)


def _to_camel_case(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


def _process_single_return_object(return_object):
    from msrest.serialization import Model
    if isinstance(return_object, Model):
        object_dict = return_object.as_dict()
        return json.dumps({_to_camel_case(k): v for k, v in object_dict.items()}, indent=4, sort_keys=True)
    elif isinstance(return_object, dict):
        return json.dumps(return_object, indent=4, sort_keys=True)
    else:
        sys.exit("CLIError: Unsupported command output type: {}".format(type(return_object)))


class dummy(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def _configure_debugging(debug_mode, func_name):
    if debug_mode:
        # Turn on diagnostics
        from azureml.core import diagnostic_log
        return diagnostic_log(context_name=func_name)
    else:
        return dummy()


def start_cli():
    """ This function is called by __main__.py to start the azureml._cli"""

    global AZUREML_CLI_IN_USE
    AZUREML_CLI_IN_USE = True

    parser = argparse.ArgumentParser(description="AzureML-specific cli")
    parser.add_argument("--debug", action="store_true", dest="debug")
    _register_subgroups_and_commands(parser)

    options = parser.parse_args()
    options_dict = vars(options)

    debug_mode = options_dict.pop('debug', False)

    # If there is a function specified to handle a cli command, then function_path is a key in the returned options.
    if "function_path" in options_dict:
        with _configure_debugging(debug_mode, options_dict["function_path"]):
            function_object = _get_command_function_object(options_dict["function_path"])
            cli_to_function_arg_map = options_dict["cli_to_function_args"]

            # Run the appropriate command handler function
            filter_function_arguments = {}

            for arg_name, arg_value in options_dict.items():
                if arg_name in ["function_path", "cli_to_function_args"]:
                    continue

                function_arg_name = cli_to_function_arg_map[arg_name]
                filter_function_arguments[function_arg_name] = arg_value

            try:
                # Calling the function to handle the command.
                # Similar to azure.cli, we just print whatever we get in return.

                # All function arguments in python are treated as both positional and keyword based.
                # So just unwrapping the dictionary here works irrespective of the order of
                # positional arguments in the handler function.

                cmd_output = function_object(**filter_function_arguments)

                # TODO: This is mainly for the project system commands.
                # This code needs to move to project system so the CLIs always get back a dictionary, rather
                # than arbitrary class objects.
                from collections import Iterator
                if cmd_output:
                    if isinstance(cmd_output, (list, Iterator)):
                        for element in cmd_output:
                            print(_process_single_return_object(element))
                    else:
                        print(_process_single_return_object(cmd_output))
            except Exception as e:
                sys.exit(e)
