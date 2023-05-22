# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" cmd_computetarget_attach.py, A file for handling compute target attach commands."""

import getpass

from azureml.core.compute import ComputeTarget
from azureml.core.compute import AdlaCompute
from azureml.core.compute import AksCompute
from azureml.core.compute import BatchCompute
from azureml.core.compute import DataFactoryCompute
from azureml.core.compute import DatabricksCompute
from azureml.core.compute import HDInsightCompute
from azureml.core.compute import RemoteCompute
from azureml.core.compute import KubernetesCompute
from azureml._base_sdk_common.common import CLICommandOutput, RUNCONFIGURATION_EXTENSION, COMPUTECONTEXT_EXTENSION
from azureml.exceptions import UserErrorException
from ._common import get_workspace_or_default


# pylint: disable=line-too-long

""" Modules """


def attach_remote(name, address, ssh_port, username, password='', private_key_file='',
                  private_key_passphrase='', workspace_name=None, resource_group_name=None, ):
    workspace = get_workspace_or_default(workspace_name=workspace_name, resource_group=resource_group_name)

    print('Attaching compute resource...')
    attach_config = RemoteCompute.attach_configuration(username=username, address=address, ssh_port=ssh_port,
                                                       password=password, private_key_file=private_key_file,
                                                       private_key_passphrase=private_key_passphrase)
    ComputeTarget.attach(workspace, name, attach_config)
    print('Resource attach submitted successfully.')
    print('To see if your compute target is ready to use, run:')
    print('  az ml computetarget show -n {}'.format(name))


def attach_hdi(name, address, ssh_port, username, password='', private_key_file='',
               private_key_passphrase='', workspace_name=None, resource_group_name=None):
    workspace = get_workspace_or_default(workspace_name=workspace_name, resource_group=resource_group_name)

    print('Attaching hdi compute cluster...')
    attach_config = HDInsightCompute.attach_configuration(username=username, address=address, ssh_port=ssh_port,
                                                          password=password, private_key_file=private_key_file,
                                                          private_key_passphrase=private_key_passphrase)
    ComputeTarget.attach(workspace, name, attach_config)
    print('HDI cluster compute attach request submitted successfully.')
    print('To see if your compute target is ready to use, run:')
    print('  az ml computetarget show -n {}'.format(name))


def attach_aks(name, compute_resource_id, workspace_name=None, resource_group_name=None):
    _attach_compute_internal(name, compute_resource_id, AksCompute, workspace_name, resource_group_name)


def attach_datafactory(name, compute_resource_id, workspace_name=None, resource_group_name=None):
    _attach_compute_internal(name, compute_resource_id, DataFactoryCompute, workspace_name, resource_group_name)


def attach_databricks(name, access_token, compute_resource_id, workspace_name=None, resource_group_name=None):
    workspace = get_workspace_or_default(workspace_name, resource_group_name)

    print('Attaching compute resource...')
    attach_config = DatabricksCompute.attach_configuration(resource_id=compute_resource_id, access_token=access_token)
    ComputeTarget.attach(workspace, name, attach_config)
    print('Resource attach submitted successfully.')
    print('To see if your compute target is ready to use, run:')
    print('  az ml computetarget show -n {}'.format(name))


def attach_adla(name, compute_resource_id, workspace_name=None, resource_group_name=None):
    _attach_compute_internal(name, compute_resource_id, AdlaCompute, workspace_name, resource_group_name)


def attach_batch(name, compute_resource_id, workspace_name=None, resource_group_name=None):
    _attach_compute_internal(name, compute_resource_id, BatchCompute, workspace_name, resource_group_name)


def attach_kubernetes(name, compute_resource_id, workspace_name=None, resource_group_name=None, namespace=None):
    workspace = get_workspace_or_default(workspace_name=workspace_name, resource_group=resource_group_name)

    print('Attaching kubernetes compute...')
    k8s_attach_configuration = KubernetesCompute.attach_configuration(resource_id=compute_resource_id,
                                                                      namespace=namespace)
    ComputeTarget.attach(workspace, name, k8s_attach_configuration)

    print('Resource attach submitted successfully.')
    print('To see if your compute target is ready to use, run:')
    print('  az ml computetarget show -n {}'.format(name))


def _attach_compute_internal(name, compute_resource_id, compute_type, workspace_name=None, resource_group_name=None):
    workspace = get_workspace_or_default(workspace_name=workspace_name, resource_group=resource_group_name)

    print('Attaching compute resource...')
    attach_config = compute_type.attach_configuration(resource_id=compute_resource_id)
    ComputeTarget.attach(workspace, name, attach_config)
    print('Resource attach submitted successfully.')
    print('To see if your compute target is ready to use, run:')
    print('  az ml computetarget show -n {}'.format(name))


# The function takes an input from a user.
# prompt_message denotes a string, which is printed at the command prompt before
# a user enters the information.
# hide=False means the entered text will be echoed on the terminal.
# The function returns the entered text or raises an error in case of an incorrect input.
def _get_user_input(prompt_message, hide=False, allow_empty=False):
    if hide:
        return _password_input(prompt_message, allow_empty)
    else:
        return _text_input(prompt_message, allow_empty)


# The function takes a password as input from the current terminal.
# Takes as input a string that it displays to a user for entering a password.
# Prompts a user for password two times so that any password entering typos can be reduced.
def _password_input(prompt_message, allow_empty=False):
    password_1 = getpass.getpass(prompt_message)
    if len(password_1) <= 0 and not allow_empty:
        raise UserErrorException("Empty password not allowed. Please try again.")

    password_2 = getpass.getpass("Re-enter the password for confirmation:")
    if password_1 == password_2:
        return password_1
    else:
        raise UserErrorException("Entered passwords don't match. Please try again.")


# The function takes a text as input from the current terminal.
# Takes as input a string that it displays to a user for entering a text.
# Prompts a user for a text two times so that any text entering typos can be reduced.
# The text that user enters is displayed on the terminal.
def _text_input(prompt_message, allow_empty=False):
    text_1 = input(prompt_message)
    if len(text_1) <= 0 and not allow_empty:
        raise UserErrorException("Empty value not allowed. Please try again.")

    text_2 = input("Re-enter the value for confirmation:")
    if text_1 == text_2:
        return text_1
    else:
        raise UserErrorException("Entered values don't match. Please try again.")


# Prints the status messages on the terminal for the compute_target attach command before returning to user.
def _get_attach_status(compute_target_name, prepare_required=True):
    """
    Returns attach status as an object of CLICommandOutput.
    :param compute_target_name:
    :param prepare_required:
    :return: an object of CLICommandOutput, which contains the status.
    :rtype: CLICommandOutput
    """
    command_output = CLICommandOutput("")
    command_output.append_to_command_output("Successfully created the following files:")

    command_output.append_to_command_output("{name}{compute_extension}: contains connection and configuration "
                                            "information for a remote execution "
                                            "target".format(name=compute_target_name,
                                                            compute_extension=COMPUTECONTEXT_EXTENSION))

    command_output.append_to_command_output(
        "{name}{runconfiguration_extension}: set of run options used when executing within the Azure ML "
        "Workbench application".format(name=compute_target_name,
                                       runconfiguration_extension=RUNCONFIGURATION_EXTENSION))

    if prepare_required:
        command_output.append_to_command_output("")
        command_output.append_to_command_output("Before running against {name}, you need to prepare it with "
                                                "your project's environment by "
                                                "running:".format(name=compute_target_name))

        command_output.append_to_command_output("az ml experiment prepare -c {name}".format(name=compute_target_name))

    command_output.set_do_not_print_dict()
    return command_output
