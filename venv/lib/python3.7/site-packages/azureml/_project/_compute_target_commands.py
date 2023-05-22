# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" _compute_target_commands.py, A file for handling compute target logic for a project."""

from __future__ import print_function

import collections
import os
import uuid

import requests
import azureml._vendor.ruamel.yaml as ruamelyaml

from azureml._base_sdk_common.common import COMPUTECONTEXT_EXTENSION, get_run_config_dir_name
from azureml._base_sdk_common.credentials import set_credentials, remove_credentials
from azureml._base_sdk_common.execution_service_address import ExecutionServiceAddress
from azureml.exceptions import ComputeTargetException
from azureml.core.runconfig import RunConfiguration

# These may need to be moved to base_sdk_common.

# The prefix in the password field in the <target>.compute file
_SECRET_PREFIX = "AzureMlSecret="


# TODO: This file has some prints and interactive inputs, which may not be suitable for SDK
# But, will be fixed in a later PR.

# TODO: This file also returns CLICommandOutput for most functions, which needs to change.

SSHKeyInfo = collections.namedtuple("SSHKeyInfo", ["private_key_file", "passphrase"])


def _get_ssh_info(compute_target_object):
    ssh_key_information = None
    if compute_target_object.use_ssh_keys:
        ssh_key_information = SSHKeyInfo(compute_target_object.private_key_file,
                                         compute_target_object.private_key_passphrase)

    return ssh_key_information


def attach_ssh_based_compute_targets(experiment, source_directory, compute_target_object):
    """
    Attaches a SSH based compute target to a project.
    :param experiment:
    :type experiment: azureml.core.experiment.Experiment
    :param compute_target_object:
    :type compute_target_object: azureml.core.compute.target._SSHBasedComputeTarget
    :return: None if successful, throws an exception otherwise.
    """
    compute_skeleton = compute_target_object._serialize_to_dict()
    ssh_key_info = _get_ssh_info(compute_target_object)
    _set_password_or_key_info(compute_target_object.name, compute_target_object.address,
                              compute_target_object.username, compute_target_object.password,
                              ssh_key_info, experiment, compute_skeleton)

    # Writing compute and runconfig files.
    _write_compute_run_config(source_directory, compute_target_object, compute_skeleton)


def attach_batchai_compute_target(experiment, source_directory, compute_target_object):
    """
    Attaches BatchAI compute target to a project.
    :param experiment:
    :type experiment: azureml.core.experiment.Experiment
    :param compute_target_object:
    :type compute_target_object: azureml.core.compute.target._BatchAITarget
    :return: None if successful, throws an exception otherwise.
    """
    compute_skeleton = compute_target_object._serialize_to_dict()
    # Writing compute and runconfig files.
    _write_compute_run_config(source_directory, compute_target_object, compute_skeleton)


def detach_compute_target(experiment, source_directory, compute_target_name):
    """
    Detach a compute target from the project.
    :param experiment:
    :type experiment: azureml.core.experiment.Experiment
    :param compute_target_name:
    :type compute_target_name: str
    :return: None if detach is successful, throws an exception otherwise.
    """

    # Bug fixed: We change the directory here to the aml_config directory, using the
    # user specified relative path. Then, in delete function also we try to change the
    # directory to the project directory using the user specified relative path, which is a
    # a bug and breaks. So, converting the user specified relative path to the absolute path
    # and will pass that to the delete function.

    run_config_dir_name = get_run_config_dir_name(source_directory)
    full_file_path = os.path.join(source_directory,
                                  run_config_dir_name, compute_target_name + COMPUTECONTEXT_EXTENSION)
    # Find username before deleting
    with open(full_file_path, 'r') as outfile:
        data = ruamelyaml.load(outfile)

    for key, value in data.items():
        if key == 'password':
            if value.startswith(_SECRET_PREFIX):
                remove_credentials(experiment, value[len(_SECRET_PREFIX):])

    os.remove(full_file_path)


def get_all_compute_target_objects(project_object):
    """
    Returns a dictionary, where key is the compute target name,
    and value is the compute target type.
    :param project_object:
    :type project_object: azureml.core.project.Project
    :return:
    :rtype: dict
    """
    compute_configs_dict = dict()

    run_config_dir_name = get_run_config_dir_name(project_object.project_directory)
    for _, _, filenames in os.walk(os.path.join(project_object.project_directory,
                                                run_config_dir_name)):
        for file in filenames:
            file_extension = os.path.splitext(file)[-1]
            compute_config_name = os.path.splitext(file)[0]
            if file_extension == COMPUTECONTEXT_EXTENSION:
                compute_file_path = os.path.join(project_object.project_directory,
                                                 run_config_dir_name, file)
                if os.path.isfile(compute_file_path):
                    with open(compute_file_path, "r") as compute_config:
                        compute_config_dict = ruamelyaml.load(compute_config)
                        try:
                            compute_configs_dict[compute_config_name] = compute_config_dict["type"]
                        except ComputeTargetException:
                            # TODO: Skipping the error compute configs.
                            pass

    return compute_configs_dict


# Sets the password or key info in the compute skeleton.
# If needed, looks up the workspace public key info from the
# execution service.
def _set_password_or_key_info(name, address, username, password, ssh_key_info, experiment, compute_skeleton):
    import paramiko
    from azureml.core.compute_target import _SSHBasedComputeTarget
    if not ssh_key_info:
        # Credential management for password.
        key = name + '#' + username + '#' + uuid.uuid4().hex
        set_credentials(experiment, key, password)
        compute_skeleton[_SSHBasedComputeTarget._PASSWORD_KEY] = _SECRET_PREFIX + key
    else:
        # Credential management for SSH key case.
        response_json = _get_workspace_key(experiment)

        port = 22
        address_minus_port = address
        if ":" in address:
            address_minus_port, port = address.split(":")
            port = int(port)

        private_key = None
        if ssh_key_info and ssh_key_info.private_key_file:
            # TODO: We only support RSA keys for now. Investigate DSA
            private_key = paramiko.RSAKey.from_private_key_file(ssh_key_info.private_key_file,
                                                                password=ssh_key_info.passphrase)

        try:
            # These print statement will be printed even in azureml._cli.
            # TODO: Need to remove these prints for SDK.
            print("Installing AML Workbench public key on remote host.")
            _install_public_ssh_key(address_minus_port,
                                    port,
                                    username,
                                    response_json["publicKey"],
                                    private_key=private_key)
        except (paramiko.ssh_exception.AuthenticationException, paramiko.ssh_exception.SSHException) as exception:
            print("Error installing public key: %s" % str(exception))
            print("Please install the public key below on your compute target manually, or correct the error "
                  "and try again.")

            print("\nPublic key: ")
            print(response_json["publicKey"])
            print("\n")
            print("Please append the public key in ~/.ssh/authorized_keys file on the attached compute target. ")

        compute_skeleton[_SSHBasedComputeTarget._PRIVATE_KEY_KEY] = response_json["privateKey"]


def _install_public_ssh_key(address, port, username, public_key_to_install, private_key=None):
    # Check if the key is already installed
    authorized_keys_file = "~/.ssh/authorized_keys"
    grep_command = "grep '%s' %s -c -m 1" % (public_key_to_install.strip(), authorized_keys_file)

    (grep_out, grep_err) = _execute_ssh_command(address, port, username, grep_command, private_key=private_key)

    # If not installed, install it
    if int(grep_out[0]) < 1:
        install_command = 'echo \"' + public_key_to_install + '" >> ' + authorized_keys_file
        _execute_ssh_command(address, port, username, install_command, private_key=private_key)


# Writes the compute config and run config for a target in a project.
def _write_compute_run_config(source_directory, compute_target_object, compute_yaml):
    """
    :param source_directory:
    :type source_directory: str
    :param compute_target_object:
    :type compute_target_object: azureml.core.compute_target.AbstractComputeTarget
    :param compute_yaml:
    :type compute_yaml: dict
    :return:
    """
    from azureml.core.compute_target import _BatchAITarget
    # Writing the target.compute file.
    run_config_dir_name = get_run_config_dir_name(source_directory)
    file_path = os.path.join(source_directory, run_config_dir_name,
                             compute_target_object.name + COMPUTECONTEXT_EXTENSION)
    with open(file_path, 'w') as outfile:
        ruamelyaml.dump(compute_yaml, outfile, default_flow_style=False)

    # This creates a run config and writes it in the aml_config/<compute_target_name>.runconfig file
    run_config_object = RunConfiguration()
    run_config_object.target = compute_target_object

    if compute_target_object.type == _BatchAITarget._BATCH_AI_TYPE:
        run_config_object.environment.docker.enabled = True

    run_config_object.framework = compute_target_object._default_framework

    run_config_object.save(name=compute_target_object.name, path=source_directory)


# Connect to the specified server and run the given command.
# This method will try mutliple authentication methods until one success.
# First, if given, it will use the private key.
# Then, it will try keys from the users ssh_agent (if any)
# Finally it will try password authentication (if a password is supplied).
# Note that currently the command line option to use a password to install keys
# is disabled, so the password case will never be reached.
# If no authentication methods succeed, it will raie paramiko.ssh_exception.AuthenticationException
def _execute_ssh_command(address, port, username, command_to_run, password=None, private_key=None):
    import paramiko

    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(address, port, username, pkey=private_key, password=password)

    stdin, stdout, stderr = ssh.exec_command(command_to_run)

    return stdout.readlines(), stderr.readlines()


# This function contacts the execution service and gets the public key and the credential
# service lookup key.
def _get_workspace_key(experiment):
    cloud_execution_service_address = experiment.workspace.service_context._get_run_history_url()
    execution_service_details = ExecutionServiceAddress(cloud_execution_service_address)
    experiment_uri_path = experiment.workspace.service_context._get_experiment_scope(experiment.name)
    uri = execution_service_details.address
    uri += "/execution/v1.0" + experiment_uri_path + "/getorcreateworkspacesshkey"

    auth_header = experiment.workspace._auth_object.get_authentication_header()
    headers = {}

    headers.update(auth_header)
    response = requests.post(uri, headers=headers)

    if response.status_code >= 400:
        from azureml._base_sdk_common.common import get_http_exception_response_string
        raise ComputeTargetException(get_http_exception_response_string(response))

    return response.json()
