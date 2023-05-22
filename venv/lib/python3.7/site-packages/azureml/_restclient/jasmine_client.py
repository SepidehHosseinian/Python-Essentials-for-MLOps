# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Access jasmineclient"""
from .experiment_client import ExperimentClient
from .models.many_models_run_validation_output import ManyModelsRunValidationOutput


class JasmineClient(ExperimentClient):
    """
    Jasmine APIs

    :param host: The base path for the server to call.
    :type host: str
    :param auth: Client authentication
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :param resource_group_name:
    :type resource_group_name: str
    :param workspace_name:
    :type workspace_name: str
    :param project_name:
    :type project_name: str
    """

    def __init__(self,
                 service_context,
                 experiment_name,
                 experiment_id,
                 **kwargs):
        """
        Constructor of the class.
        """
        try:
            from azureml.train.automl import \
                __version__ as azureml_train_automl
        except Exception:
            azureml_train_automl = None
        try:
            from azureml.train.automl.runtime import \
                __version__ as azureml_train_automl_runtime
        except Exception:
            azureml_train_automl_runtime = None

        self.automl_user_agent = "azureml.train.automl/{} azureml.train.automl.runtime/{}"\
            .format(azureml_train_automl, azureml_train_automl_runtime)

        super(JasmineClient, self).__init__(
            service_context, experiment_name, experiment_id, **kwargs)

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        if user_agent is None or not user_agent.__contains__("automl"):
            user_agent = "{} {}".format(user_agent, self.automl_user_agent)
        return self._service_context._get_jasmine_restclient(user_agent=user_agent)

    def post_validate_service(self, create_parent_run_dto):
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.validate_run_input, create_parent_run_dto)

    def post_on_demand_model_explain_run(self, run_id, model_explain_dto):
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.on_demand_model_explain, run_id, model_explain_dto)

    def post_on_demand_model_test_run(self, run_id, model_test_dto):
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.on_demand_model_test, run_id, model_test_dto)

    def post_parent_run(self, create_parent_run_dto=None):
        """
        Post a new parent run to Jasmine
        :param create_parent_run_dto:
        :return:
        """
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.create_parent_run_method, create_parent_run_dto)

    def post_remote_jasmine_snapshot_run(self, parent_run_id, run_definition, snapshotId):
        """
        Post a new experiment to Jasmine
        :param parent_run_id:
        :type parent_run_id: str
        :param run_definition: run configuration information
        :type run_definition: ~_restclient.models.RunDefinition
        :param snapshotId: id of the snapshot containing the project files
        :return:
        """
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.submit_remote_snapshot_run, parent_run_id, run_definition, snapshotId)

    def get_next_task(self, parent_run_id, local_history_payload, start_child_run=False):
        """
        Get task for next iteration to run from Jasmine
        :param parent_run_id:
        :type parent_run_id: str
        :param local_history_payload:
        :type local_history_payload: ~_restclient.models.LocalRunGetNextTaskInput
        :param start_child_run: whether to start child run
        :type start_child_run: bool
        :return:
        """
        dto = self._execute_with_experimentid_arguments(
            self._client.jasmine.local_run_get_next_task,
            parent_run_id,
            start_child_run,
            local_history_payload
        )
        return dto if dto.__dict__ is not None else None

    def get_next_task_batch(self, parent_run_id, local_run_get_next_task_batch_input, start_child_runs=False):
        """
        Get batch of next tasks / iterations to run from Jasmine
        :param parent_run_id:
        :type parent_run_id: str
        :param local_run_get_next_task_batch_input:
        :type local_run_get_next_task_batch_input: ~_restclient.models.LocalRunGetNextTaskBatchInput
        :param start_child_runs: whether to start child runs
        :type start_child_runs: bool
        :return:
        """
        local_run_get_next_task_batch_input.version = 1
        dto = self._execute_with_experimentid_arguments(
            self._client.jasmine.local_run_get_next_task_batch,
            parent_run_id,
            start_child_runs,
            local_run_get_next_task_batch_input)
        return dto if dto.__dict__ is not None else None

    def get_next_pipeline(self, parent_run_id, worker_id):
        """
        Get next set of pipelines to run from Jasmine
        :param parent_run_id:
        :type parent_run_id: str
        :param worker_id:
        :type worker_id: str
        :return:
        """
        dto = self._execute_with_experimentid_arguments(
            self._client.jasmine.get_pipeline, parent_run_id, worker_id)
        return dto if dto.__dict__ is not None else None

    def set_parent_run_status(self, parent_run_id, target_status):
        """
        Post a new experiment to Jasmine
        :param parent_run_id:
        :type parent_run_id: str
        :param target_status:
        :type target_status: str
        :return:
        """

        return self._execute_with_experimentid_arguments(
            self._client.jasmine.change_run_status,
            parent_run_id, target_status)

    def cancel_run(self, run_id):
        """
        Post a new experiment to Jasmine
        :param run_id:
        :type run_id: str
        :return:
        """
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.cancel_run,
            run_id)

    def continue_remote_run(self, run_id, iterations=None, exit_time_sec=None, exit_score=None):
        """
        Post a new experiment to Jasmine
        :param child_run_id:
        :type child_run_id: str
        :return:
        """
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.continue_run,
            run_id, iterations, exit_time_sec, exit_score)

    def get_feature_profiles(self, run_id, feature_input_dto):
        """
        get feature profiles from Jasmine
        :param run_id:
        :type run_id: str
        :param feature_profile_input_dto:
        :type feature_profile_input_dto:
         ~_restclient.models.FeatureProfileInputDto
        :return:
        """
        return self._execute_with_experimentid_arguments(
            self._client.jasmine.get_feature_profiles,
            run_id,
            feature_input_dto)

    def get_curated_environment(self, scenario, enable_dnn, enable_gpu, compute, compute_sku, label_override=None):
        """
        get automl curated environment for specified configuration
        :param scenario:
        :type scenario: str
        :param enable_dnn:
        :type bool:
        :param enable_gpu:
        :type bool:
        :param compute:
        :type str:
        :param compute_sku
        :type str:
        :param label_override
        :type str:
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        from azureml._restclient.models import EnvironmentDefinition
        from azureml.core import Environment
        from msrest import Serializer

        from .models import AutoMLCuratedEnvInput

        automl_curated_env_input = AutoMLCuratedEnvInput(version=1,
                                                         enable_dnn=enable_dnn,
                                                         enable_gpu=enable_gpu,
                                                         scenario=scenario,
                                                         compute=compute,
                                                         compute_sku=compute_sku,
                                                         label_override=label_override)

        automl_curated_env_output = self._execute_with_experimentid_arguments(
            self._client.jasmine.get_curated_environment,
            automl_curated_env_input)

        # This is a workaround to convert EnvironmentDefinition => azureml.core.Environment
        client_models = {"EnvironmentDefinition": EnvironmentDefinition}
        serializer = Serializer(client_models)
        environment_json = serializer.body(automl_curated_env_output.environment, "EnvironmentDefinition")
        environment = Environment._deserialize_and_add_to_object(environment_json)

        return environment

    def validate_many_models_run_input(self, max_concurrent_runs: int, automl_settings: str,
                                       number_of_processes_per_core: int)\
            -> ManyModelsRunValidationOutput:
        """
        validate many models run input
        :param max_concurrency:
        :param automl_settings
        """
        from .models.many_models_run_validation_input import ManyModelsRunValidationInput
        input = ManyModelsRunValidationInput(
            version=1, max_concurrent_runs=max_concurrent_runs, auto_ml_settings=automl_settings,
            number_of_processes_per_core=number_of_processes_per_core)

        validation_output = self._execute_with_experimentid_arguments(
            self._client.jasmine.validate_many_models_run_input,
            input)

        return validation_output
