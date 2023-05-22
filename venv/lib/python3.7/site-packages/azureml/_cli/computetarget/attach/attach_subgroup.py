# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli import abstract_subgroup
from azureml._cli import cli_command
from azureml._cli import argument


class AttachSubGroup(abstract_subgroup.AbstractSubGroup):
    """This class defines the attach sub group."""

    def get_subgroup_name(self):
        """Returns the name of the subgroup.
        This name will be used in the cli command."""
        return "attach"

    def get_subgroup_title(self):
        """Returns the subgroup title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the subgroup."""
        return "attach subgroup commands"

    def get_nested_subgroups(self):
        """Returns sub-groups of this sub-group."""
        return super(AttachSubGroup, self).compute_nested_subgroups(__package__)

    def get_commands(self, for_azure_cli=False):
        """ Returns commands associated at this sub-group level."""
        # TODO: Adding commands to a list can also be automated, if we assume the
        # command function name to start with a certain prefix, like _command_*
        commands_list = [
            self._command_attach_remote(),
            self._command_attach_aks(),
            self._command_attach_kubernetes(),
        ]
        return commands_list

    def _command_attach_remote(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_remote"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'workspace_name=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'
        remote_username = argument.Argument("username", "--username", "-u", required=True,
                                            help="The username for the remote machine being attached. Must also "
                                                 "provide either a password or public and private key files.")
        remote_password = argument.Argument("password", "--password", "-p",
                                            help="The password for the remote machine being attached. Must either "
                                                 "provide a password or public and private key files.")
        remote_private_key_file = argument.Argument("private_key_file", "--private-key-file", "",
                                                    help="Path to a file containing the private key information for"
                                                         " the remote machine being attached.")

        return cli_command.CliCommand("remote", "Attach a remote machine without Docker as a compute target"
                                                " to the workspace.",
                                      [workspace_name, resource_group, target_name, remote_username, remote_password,
                                       remote_private_key_file, argument.PRIVATE_KEY_PASSPHRASE,
                                       argument.ADDRESS.get_required_true_copy(),
                                       argument.SSH_PORT.get_required_true_copy()], function_path)

    def _command_attach_hdi(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_hdi"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'workspace_name=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'
        hdi_username = argument.Argument("username", "--username", "-u", required=True,
                                         help="The username for the HDI being attached. Must also provide either a "
                                              "password or public and private key files.")
        hdi_password = argument.Argument("password", "--password", "-p",
                                         help="The password for the HDI being attached. Must either provide a "
                                              "password or public and private key files.")
        hdi_private_key_file = argument.Argument("private_key_file", "--private-key-file", "",
                                                 help="Path to a file containing the private key information for the "
                                                      "HDI being attached.")
        return cli_command.CliCommand("hdi", "Attach HDI cluster to the workspace.",
                                      [workspace_name, resource_group, target_name, hdi_username, hdi_password,
                                       hdi_private_key_file, argument.PRIVATE_KEY_PASSPHRASE,
                                       argument.ADDRESS.get_required_true_copy(),
                                       argument.SSH_PORT.get_required_true_copy()], function_path)

    def _command_attach_aks(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_aks"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'workspace_name=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'

        return cli_command.CliCommand("aks", "Attach an AKS cluster to the workspace.",
                                      [workspace_name, resource_group, target_name,
                                       argument.COMPUTE_RESOURCE_ID.get_required_true_copy()], function_path)

    def _command_attach_datafactory(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_datafactory"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'aml_workspace=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the data factory name. Must be unique to the workspace.'

        return cli_command.CliCommand("datafactory", "Attach a data factory to the workspace.",
                                      [workspace_name, resource_group, target_name,
                                       argument.COMPUTE_RESOURCE_ID.get_required_true_copy()], function_path)

    def _command_attach_adla(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_adla"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'aml_workspace=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'

        return cli_command.CliCommand("adla", "Attach Data Lake Analytics to the workspace.",
                                      [workspace_name, resource_group, target_name,
                                       argument.COMPUTE_RESOURCE_ID.get_required_true_copy()], function_path)

    def _command_attach_databricks(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_databricks"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'aml_workspace=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'
        access_token = \
            argument.Argument("access_token", "--access-token", "-a", required=True,
                              help='The access token for the Databricks workspace being attached. See '
                                   'this link about generating the access token: '
                                   'https://docs.databricks.com/api/latest/authentication.html#generate-a-token')

        return cli_command.CliCommand("databricks", "Attach a Databricks workspace to the workspace.",
                                      [workspace_name, resource_group, target_name, access_token,
                                       argument.COMPUTE_RESOURCE_ID.get_required_true_copy()], function_path)

    def _command_attach_batch(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_batch"
        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'aml_workspace=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'

        return cli_command.CliCommand("batch", "Attach a Batch account to the workspace.",
                                      [workspace_name, resource_group, target_name,
                                       argument.COMPUTE_RESOURCE_ID.get_required_true_copy()], function_path)

    def _command_attach_kubernetes(self):
        function_path = "azureml._base_sdk_common.cli_wrapper.cmd_computetarget_attach#attach_kubernetes"

        workspace_name = argument.WORKSPACE_NAME.clone()
        workspace_name.help = 'Name of the workspace to create this compute target under. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'workspace_name=<workspace name>". This parameter will override any set default.'
        resource_group = argument.RESOURCE_GROUP_NAME.clone()
        resource_group.help = 'Resource group corresponding to the provided workspace. A default value for all ' \
                              'commands can be set by running "az configure --defaults ' \
                              'group=<resource group name>". This parameter will override any set default.'
        target_name = argument.TARGET_NAME.get_required_true_copy()
        target_name.help = 'Specifies the compute target name. Must be unique to the workspace.'

        compute_resource_id = argument.COMPUTE_RESOURCE_ID.clone()
        compute_resource_id.required = True

        kubernetes_namespace = argument.KUBERNETES_NAMESPACE.clone()

        return cli_command.CliCommand("kubernetes",
                                      "Attach a KubernetesCompute as a compute target to the workspace.",
                                      [workspace_name, resource_group, target_name,
                                       compute_resource_id, kubernetes_namespace],
                                      function_path)
