# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli import abstract_subgroup
from azureml._cli import cli_command
from azureml._cli import argument


class PipelineSubGroup(abstract_subgroup.AbstractSubGroup):
    """This class defines the pipeline sub group."""

    def get_subgroup_name(self):
        """Returns the name of the subgroup.
        This name will be used in the cli command."""
        return "pipeline"

    def get_subgroup_title(self):
        """Returns the subgroup title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the subgroup."""
        return "pipeline subgroup commands"

    def get_nested_subgroups(self):
        """Returns sub-groups of this sub-group."""
        return super(PipelineSubGroup, self).compute_nested_subgroups(__package__)

    def get_commands(self, for_azure_cli=False):
        """ Returns commands associated at this sub-group level."""
        commands_list = [self._command_pipeline_list(),
                         self._command_pipeline_show(),
                         self._command_pipeline_enable(),
                         self._command_pipeline_disable(),
                         self._command_pipeline_list_steps(),
                         self._command_schedule_create(),
                         self._command_schedule_update(),
                         self._command_schedule_enable(),
                         self._command_schedule_disable(),
                         self._command_last_pipeline_run_show(),
                         self._command_pipeline_runs_list(),
                         self._command_schedule_show(),
                         self._command_pipeline_create(),
                         self._command_pipeline_clone(),
                         self._command_pipeline_get(),
                         self._command_pipeline_draft_show(),
                         self._command_pipeline_drafts_list(),
                         self._command_pipeline_draft_delete(),
                         self._command_pipeline_draft_submit(),
                         self._command_pipeline_draft_publish(),
                         self._command_pipeline_draft_create(),
                         self._command_pipeline_draft_clone(),
                         self._command_pipeline_draft_update()]
        return commands_list

    def _command_pipeline_list(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#list_pipelines"
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("list", "List all pipelines and respective schedules in the workspace.",
                                      [output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_show(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#show_pipeline"
        pipeline_id = argument.Argument("pipeline_id", "--pipeline-id", "-i", required=True,
                                        help="ID of the pipeline to show (guid)")
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("show", "Show details of a pipeline and respective schedules.",
                                      [pipeline_id,
                                       output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_disable(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#disable_pipeline"
        pipeline_id = argument.Argument("pipeline_id", "--pipeline-id", "-i", required=True,
                                        help="ID of the pipeline to disable (guid)")
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("disable", "Disable a pipeline from running.",
                                      [pipeline_id,
                                       output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_enable(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#enable_pipeline"
        pipeline_id = argument.Argument("pipeline_id", "--pipeline-id", "-i", required=True,
                                        help="ID of the pipeline to enable (guid)")
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("enable", "Enable a pipeline and allow it to run.",
                                      [pipeline_id,
                                       output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_list_steps(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#list_pipeline_steps"
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("list-steps", "List the step runs generated from a pipeline run",
                                      [output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME,
                                       argument.RUN_ID_OPTION.get_required_true_copy()], function_path)

    def _command_schedule_create(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#create_schedule"
        pipeline_id = argument.Argument("pipeline_id", "--pipeline-id", "-i", required=True,
                                        help="ID of the pipeline to create schedule (guid)")
        name = argument.Argument("name", "--name", "-n", required=True,
                                 help="Name of schedule")
        experiment_name = argument.Argument("experiment-name", "--experiment-name", "-e", required=True,
                                            help="Name of experiment")
        schedule_yaml = argument.Argument("schedule_yaml", "--schedule-yaml", "-y",
                                          required=False,
                                          help="Schedule  YAML input")
        return cli_command.CliCommand("create-schedule", "Create a schedule.",
                                      [name, pipeline_id, experiment_name,
                                       schedule_yaml,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_schedule_update(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#update_schedule"
        schedule_id = argument.Argument("schedule_id", "--schedule-id", "-s", required=True,
                                        help="ID of the schedule to show (guid)")
        name = argument.Argument("name", "--name", "-n", required=False,
                                 help="Name of the schedule to show (guid)")
        status = argument.Argument("status", "--status", "-t", required=False,
                                   help="Status of the schedule to show (guid)")
        schedule_yaml = argument.Argument("schedule_yaml", "--schedule-yaml", "-y",
                                          required=False,
                                          help="Schedule  YAML input")
        return cli_command.CliCommand("update-schedule", "update a schedule.",
                                      [schedule_id,
                                       name,
                                       status,
                                       schedule_yaml,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_last_pipeline_run_show(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#show_last_pipeline_run"
        schedule_id = argument.Argument("schedule_id", "--schedule-id", "-s", required=True,
                                        help="ID of the schedule to show (guid)")
        return cli_command.CliCommand("last-pipeline-run", "Show last pipeline run for a schedule.",
                                      [schedule_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_runs_list(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#list_pipeline_runs"
        schedule_id = argument.Argument("schedule_id", "--schedule-id", "-s", required=True,
                                        help="ID of the schedule to show (guid)")
        return cli_command.CliCommand("pipeline-runs-list", "List pipeline runs generated from a schedule.",
                                      [schedule_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_schedule_disable(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#disable_schedule"
        schedule_id = argument.Argument("schedule_id", "--schedule-id", "-s", required=True,
                                        help="ID of the schedule to show (guid)")
        return cli_command.CliCommand("disable-schedule", "Disable a schedule from running.",
                                      [schedule_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_schedule_enable(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#enable_schedule"
        schedule_id = argument.Argument("schedule_id", "--schedule-id", "-s", required=True,
                                        help="ID of the schedule to show (guid)")
        return cli_command.CliCommand("enable-schedule", "Enable a schedule and allow it to run.",
                                      [schedule_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_schedule_show(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#show_schedule"
        schedule_id = argument.Argument("schedule_id", "--schedule-id", "-s", required=True,
                                        help="ID of the schedule to show (guid)")
        return cli_command.CliCommand("show-schedule", "Show details of a schedule.",
                                      [schedule_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_create(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#create_pipeline"
        pipeline_yaml = argument.Argument("pipeline_yaml", "--pipeline-yaml", "-y", required=True,
                                          help="YAML file which defines a pipeline")
        name = argument.Argument("name", "--name", "-n", required=True, help="Name to assign to the pipeline")
        description = argument.Argument("description", "--description", "-d", required=False,
                                        help="Description text of the pipeline")
        version = argument.Argument("version", "--version", "-v", required=False,
                                    help="Version string of the pipeline")
        allow_continue = argument.Argument(
            "continue_on_step_failure", "--continue", "-c", required=False,
            help="Boolean flag to allow a pipeline to continue executing after a step fails")
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("create", "Create a pipeline from a yaml definition.",
                                      [pipeline_yaml,
                                       name,
                                       description,
                                       version,
                                       allow_continue,
                                       output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_clone(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#clone_pipeline_run"
        pipeline_id = argument.Argument("pipeline_run_id", "--pipeline-run-id", "-i", required=True,
                                        help="ID of the PipelineRun to clone (guid)")
        path = argument.Argument("path", "--path", "-p", required=True,
                                 help="File path to save pipeline yaml to.")
        output_file = argument.Argument("output_file", "--output-file", "-f", required=False,
                                        help="File to write output in JSON format")
        return cli_command.CliCommand("clone", "Generate yml definition describing the pipeline run,"
                                               " supported only for ModuleStep for now.",
                                      [pipeline_id,
                                       path,
                                       output_file,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_get(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#get_pipeline"
        pipeline_id = argument.Argument("pipeline_id", "--pipeline-id", "-i", required=False,
                                        help="ID of the Pipeline to get (guid)")
        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-d", required=False,
                                              help="ID of the PipelineDraft to get (guid)")
        path = argument.Argument("path", "--path", "-p", required=True,
                                 help="File path to save Pipeline yaml to.")
        return cli_command.CliCommand("get", "Generate yml definition describing the pipeline.",
                                      [pipeline_id,
                                       pipeline_draft_id,
                                       path,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_show(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#show_pipeline_draft"
        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-i", required=True,
                                              help="ID of the PipelineDraft to show (guid)")
        return cli_command.CliCommand("show-draft", "Show details of a pipeline draft.",
                                      [pipeline_draft_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_drafts_list(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#list_pipeline_drafts"
        tags = argument.Argument("tags", "--tags", "-t", required=False,
                                 help="Tags for a draft with 'key=value' syntax.",
                                 action="append", default=[])
        return cli_command.CliCommand("list-drafts", "List pipeline drafts in the workspace.",
                                      [tags,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_delete(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#delete_pipeline_draft"
        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-i", required=True,
                                              help="ID of the PipelineDraft to delete (guid)")
        return cli_command.CliCommand("delete-draft", "Delete a pipeline draft.",
                                      [pipeline_draft_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_submit(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#submit_pipeline_draft"
        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-i", required=True,
                                              help="ID of the PipelineDraft to use to submit run")
        return cli_command.CliCommand("submit-draft", "Submit a run from the pipeline draft.",
                                      [pipeline_draft_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_publish(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#publish_pipeline_draft"
        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-i", required=True,
                                              help="ID of the PipelineDraft to publish")
        return cli_command.CliCommand("publish-draft", "Publish a pipeline draft as a published pipeline.",
                                      [pipeline_draft_id,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_create(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#create_pipeline_draft"
        pipeline_yml = argument.Argument("pipeline_yml", "--pipeline-yaml", "-y", required=True,
                                         help="YAML file which defines the pipeline draft")
        name = argument.Argument("name", "--name", "-n", required=True, help="Name to assign to the pipeline draft")
        description = argument.Argument("description", "--description", "-d", required=False,
                                        help="Description text of the pipeline draft")

        pipeline_parameters = argument.Argument("pipeline_parameters", "--parameters", "", required=False,
                                                help="PipelineParameters for the draft with 'key=value' syntax.",
                                                action="append", default=[])

        properties = argument.Argument("properties", "--properties", "-p", required=False,
                                       help="Properties for the draft with 'key=value' syntax.",
                                       action="append", default=[])

        tags = argument.Argument("tags", "--tags", "-t", required=False,
                                 help="Tags for the draft with 'key=value' syntax.",
                                 action="append", default=[])

        experiment_name = argument.Argument("experiment_name", "--experiment_name", "-e", required=True,
                                            help="Experiment name for the pipeline draft")
        allow_continue = argument.Argument(
            "continue_on_step_failure", "--continue", "-c", required=False,
            help="Boolean flag to allow a pipeline to continue executing after a step fails")

        return cli_command.CliCommand("create-draft", "Create a pipeline draft from a yml definition.",
                                      [pipeline_yml,
                                       name,
                                       description,
                                       experiment_name,
                                       pipeline_parameters,
                                       allow_continue,
                                       tags,
                                       properties,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_clone(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#clone_pipeline_draft"
        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-i", required=False,
                                              help="ID of the PipelineDraft to create PipelineDraft from.")
        pipeline_run_id = argument.Argument("pipeline_run_id", "--pipeline-run-id", "-r", required=False,
                                            help="ID of the PipelineRun to create PipelineDraft from")
        experiment_name = argument.Argument("experiment_name", "--experiment-name", "-e", required=False,
                                            help="Experiment name of the specified PipelineRun")
        pipeline_id = argument.Argument("pipeline_id", "--pipeline-id", "-p", required=False,
                                        help="ID of the PublishedPipeline to create PipelineDraft from")

        return cli_command.CliCommand("clone-draft", "Create a pipeline draft from an existing pipeline.",
                                      [pipeline_draft_id,
                                       pipeline_run_id,
                                       pipeline_id,
                                       experiment_name,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)

    def _command_pipeline_draft_update(self):
        function_path = "azureml.pipeline._cli_wrapper.cmd_pipeline#update_pipeline_draft"

        pipeline_draft_id = argument.Argument("pipeline_draft_id", "--pipeline-draft-id", "-i", required=False,
                                              help="ID of the PipelineDraft to update.")
        pipeline_yml = argument.Argument("pipeline_yml", "--pipeline-yaml", "-y", required=False,
                                         help="YAML file which defines the pipeline draft")
        name = argument.Argument("name", "--name", "-n", required=False, help="Name to assign to the pipeline draft")
        description = argument.Argument("description", "--description", "-d", required=False,
                                        help="Description text of the pipeline draft")

        pipeline_parameters = argument.Argument("pipeline_parameters", "--parameters", "", required=False,
                                                help="PipelineParameters for the draft with 'key=value' syntax.",
                                                action="append", default=[])

        experiment_name = argument.Argument("experiment_name", "--experiment_name", "-e", required=False,
                                            help="Experiment name for the pipeline draft")
        allow_continue = argument.Argument(
            "continue_on_step_failure", "--continue", "-c", required=False,
            help="Boolean flag to allow a pipeline to continue executing after a step fails")

        tags = argument.Argument("tags", "--tags", "-t", required=False,
                                 help="Tags for the draft with 'key=value' syntax.",
                                 action="append", default=[])

        return cli_command.CliCommand("update-draft", "Update a pipeline draft.",
                                      [pipeline_draft_id,
                                       pipeline_yml,
                                       name,
                                       description,
                                       experiment_name,
                                       tags,
                                       pipeline_parameters,
                                       allow_continue,
                                       argument.RESOURCE_GROUP_NAME,
                                       argument.WORKSPACE_NAME], function_path)
