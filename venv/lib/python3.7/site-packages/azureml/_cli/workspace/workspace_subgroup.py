# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli import abstract_subgroup
from azureml._cli import cli_command
from azureml._cli.workspace.private_endpoint.private_endpoint_arguments import argument, PE_NAME_ARGUMENT, \
    PE_RESOURCE_GROUP_ARGUMENT, PE_VNET_NAME_ARGUMENT, PE_SUBNET_NAME_ARGUMENT, PE_SUBSCRIPTION_ID_ARGUMENT, \
    PE_AUTO_APPROVAL_ARGUMENT


class WorkspaceSubGroup(abstract_subgroup.AbstractSubGroup):
    """This class defines the project sub group."""

    def get_subgroup_name(self):
        """Returns the name of the subgroup.
        This name will be used in the cli command."""
        return "workspace"

    def get_subgroup_title(self):
        """Returns the subgroup title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the subgroup."""
        return "workspace subgroup commands"

    def get_nested_subgroups(self):
        """Returns sub-groups of this sub-group."""
        return super(WorkspaceSubGroup, self).compute_nested_subgroups(__package__)

    def get_commands(self, for_azure_cli=False):
        """ Returns commands associated at this sub-group level."""
        # TODO: Adding commands to a list can also be automated, if we assume the
        # command function name to start with a certain prefix, like _command_*
        auto_registered_commands = super(WorkspaceSubGroup, self).get_commands()
        commands_list = auto_registered_commands + \
            [
                self._command_workspace_create(),
                self._command_workspace_list()
            ]
        return commands_list

    def _command_workspace_create(self):
        function_path = "azureml._cli.workspace.workspace_commands#create_workspace"

        workspace_friendly_name = argument.FRIENDLY_NAME.clone()
        workspace_friendly_name.help = "Friendly name for this workspace."

        key_vault = argument.Argument("key_vault",
                                      "--keyvault",
                                      "",
                                      help="Key Vault to be associated with this workspace.")
        storage_argument = argument.Argument("storage_account",
                                             "--storage-account",
                                             "",
                                             help="Storage account to be associated with this workspace.")
        container_registry_argument = argument.Argument(
            "container_registry",
            "--container-registry",
            "",
            help="Container Registry to be associated with this workspace.")
        adb_workspace_argument = argument.Argument(
            "adb_workspace",
            "--adb-workspace",
            "",
            help="Adb Workspace to be linked with this workspace.")
        primary_user_assigned_identity_argument = argument.Argument(
            "primary_user_assigned_identity",
            "--primary-user-assigned-identity",
            "",
            help="The resourceId of the user assigned identity that used to represent workspace identity.")
        application_insights_argument = argument.Argument(
            "app_insights",
            "--application-insights",
            "",
            help="Application Insights to be associated with this workspace.")
        cmk_keyvault_argument = argument.Argument(
            "cmk_keyvault",
            "--cmk-keyvault",
            "",
            help="The key vault containing the customer managed key in the Azure resource ID format.")
        resource_cmk_uri_argument = argument.Argument(
            "resource_cmk_uri",
            "--resource-cmk-uri",
            "",
            help="The key URI of the customer managed key to encrypt the data at rest.")
        user_assigned_identity_for_cmk_encryption_argument = argument.Argument(
            "user_assigned_identity_for_cmk_encryption",
            "--user-assigned-identity-for-cmk-encryption",
            "",
            help="The resourceId of the user assigned identity that needs to be used to access the encryption key.")
        hbi_workspace_argument = argument.Argument(
            "hbi_workspace",
            "--hbi-workspace",
            "",
            action="store_true",
            help="Specifies whether the customer data is of High Business Impact(HBI), i.e., "
                 "contains sensitive business information.")
        create_resource_group_argument = argument.Argument(
            "create_resource_group",
            "--yes",
            "-y",
            action="store_true",
            help="Create a resource group for this workspace.")
        exist_ok_workspace_argument = argument.Argument(
            "exist_ok",
            "--exist-ok",
            "",
            action="store_true",
            help="Do not fail if workspace exists.")
        sku_argument = argument.Argument(
            "sku",
            "--sku",
            "",
            default="basic",
            help="SKU/edition of a workspace -can be a 'basic' workspace or a feature rich 'enterprise' workspace.")

        tags_argument = argument.Argument(
            "tags",
            "--tags",
            "-t",
            default="",
            help="Tags associated with this workspace with 'key=value' syntax.")

        system_datastores_auth_mode_argument = argument.Argument(
            "system_datastores_auth_mode",
            "--system_datastores_auth_mode",
            "",
            action="store_true",
            default="accessKey",
            help="Determines whether or not to use credentials for the system datastores of the workspace\
                'workspaceblobstore' and 'workspacefilestore'. The default value is 'accessKey', in which case,\
                the workspace will create the system datastores with credentials.  If set to 'identity',\
                the workspace will create the system datastores with no credentials."
        )
        v1_legacy_mode_argument = argument.Argument(
            "v1_legacy_mode",
            "--v1-legacy-mode",
            "-v",
            help="Prevent using v2 API service on public Azure Resource Manager if you set this parameter true. \
                Learn more at aka.ms/amlv2network")

        return cli_command.CliCommand("create", "Create a workspace",
                                      [argument.WORKSPACE_NAME.get_required_true_copy(),
                                       workspace_friendly_name,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.LOCATION,
                                       storage_argument,
                                       key_vault,
                                       application_insights_argument,
                                       container_registry_argument,
                                       adb_workspace_argument,
                                       primary_user_assigned_identity_argument,
                                       cmk_keyvault_argument,
                                       resource_cmk_uri_argument,
                                       user_assigned_identity_for_cmk_encryption_argument,
                                       hbi_workspace_argument,
                                       create_resource_group_argument,
                                       exist_ok_workspace_argument,
                                       sku_argument,
                                       tags_argument,
                                       PE_NAME_ARGUMENT,
                                       PE_VNET_NAME_ARGUMENT,
                                       PE_SUBNET_NAME_ARGUMENT,
                                       PE_SUBSCRIPTION_ID_ARGUMENT,
                                       PE_RESOURCE_GROUP_ARGUMENT,
                                       PE_AUTO_APPROVAL_ARGUMENT,
                                       system_datastores_auth_mode_argument,
                                       v1_legacy_mode_argument,
                                       ], function_path)

    def _command_workspace_list(self):
        function_path = "azureml._cli.workspace.workspace_commands#list_workspace"

        return cli_command.CliCommand("list", "List workspaces.",
                                      [argument.RESOURCE_GROUP_NAME], function_path)
