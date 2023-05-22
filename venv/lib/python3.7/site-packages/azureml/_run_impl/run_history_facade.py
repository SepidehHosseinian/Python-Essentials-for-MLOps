# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
# Start Temp request solution
import os
import traceback
import uuid

import requests
import six
from azureml._history.utils.context_managers import TrackFolders
from azureml._async.worker_pool import WorkerPool
from azureml._logging.chained_identity import ChainedIdentity
from azureml._restclient.artifacts_client import ArtifactsClient
from azureml._restclient.assets_client import AssetsClient
from azureml._restclient.clientbase import ClientBase
from azureml._restclient.exceptions import ServiceException
from azureml._restclient.experiment_client import ExperimentClient
from azureml._restclient.metrics_client import MetricsClient
from azureml._restclient.models.create_run_dto import CreateRunDto
from azureml._restclient.run_client import RunClient
from azureml._restclient.snapshots_client import SnapshotsClient
from azureml._restclient.workspace_client import WorkspaceClient

from azureml._restclient.utils import create_session_with_retry
from azureml.core._metrics import (ArtifactBackedMetric, ScalarMetric, ListMetric, RowMetric,
                                   TableMetric, ConfusionMatrixMetric, AccuracyTableMetric,
                                   ResidualsMetric, PredictionsMetric, ImageMetric)
from azureml.core.model import Model
from azureml._common._core_user_error.user_error import (CreateChildrenFailed, StartChildrenFailed,
                                                         TwoInvalidParameter, TwoInvalidArgument)
from azureml._common._error_definition import AzureMLError
from azureml.exceptions import AzureMLException, UserErrorException, ModelPathNotFoundException, WebserviceException
from azureml.exceptions import SnapshotException

STATUS_KEY = "status"
RUN_NAME_DELIM = ">"

module_logger = logging.getLogger(__name__)

_use_v2_metrics = os.getenv('AZUREML_METRICS_V2', True)


class PropertyKeys(object):
    SNAPSHOT_ID = "ContentSnapshotId"


class RunHistoryFacade(ChainedIdentity):
    _worker_pool = None

    def __init__(self, experiment, run_id, origin, run_dto=None, user_agent=None,
                 worker_pool=None, outputs=None, py_wd=None, deny_list=None,
                 _batch_upload_metrics=True, flush_eager=False, **kwargs):
        """
        :param experiment: The experiment object.
        :type experiment: azureml.core.exepriment.Experiment
        :param run_id:
        :type run_id: str
        :param origin:
        :type origin: str
        :param run_dto:
        :type run_dto: azureml._restclient.models.create_run_dto.CreateRunDto
        :param worker_pool:
        :type worker_pool: azureml._async.worker_pool.WorkerPool
        :param user_agent:
        :type user_agent: str
        :param data_container_id:
        :type data_container_id: str
        """
        super(RunHistoryFacade, self).__init__(**kwargs)

        # deny_list is empty if not specified
        deny_list = [] if deny_list is None else deny_list

        self._experiment = experiment
        self._origin = origin
        self._run_id = run_id

        self.worker_pool = worker_pool if worker_pool is not None else RunHistoryFacade._get_worker_pool()
        base_kwargs = {"user_agent": user_agent, "worker_pool": worker_pool, "_parent_logger": self._logger}

        self.run = RunClient(self._experiment.workspace.service_context, self._experiment.name, self._run_id,
                             experiment_id=self._experiment.id, **base_kwargs)

        self.assets = AssetsClient(self._experiment.workspace.service_context, **base_kwargs)

        self.artifacts = ArtifactsClient(self._experiment.workspace.service_context, **base_kwargs)

        self.snapshots = SnapshotsClient(self._experiment.workspace.service_context, **base_kwargs)

        self.metrics = MetricsClient(self._experiment.workspace.service_context, self._experiment.name, self._run_id,
                                     use_batch=_batch_upload_metrics, flush_eager=flush_eager, **base_kwargs)
        self.run_dto = run_dto if run_dto is not None else self.run.get_run()
        self.output_file_tracker = TrackFolders(py_wd, self.artifacts, self._data_container_id, outputs, deny_list)

    @classmethod
    def _get_worker_pool(cls):
        if cls._worker_pool is None:
            cls._worker_pool = WorkerPool(_parent_logger=module_logger)
            module_logger.debug("Created a static thread pool for {} class".format(cls.__name__))
        else:
            module_logger.debug("Access an existing static threadpool for {} class".format(cls.__name__))
        return cls._worker_pool

    @property
    def _data_container_id(self):
        return getattr(self.run_dto, "data_container_id", None)

    @staticmethod
    def target_name():
        return "local"

    @staticmethod
    def create_run(
        experiment,
        name=None,
        run_id=None,
        properties=None,
        tags=None,
        typev2=None,
        display_name=None,
        description=None
    ):
        """
        :param experiment:
        :type experiment: azureml.core.experiment.Experiment
        :param name:
        :param run_id:
        :param properties:
        :param tags:
        :return:
        """
        run_id = RunHistoryFacade.create_run_id(run_id)
        client = RunClient(
            experiment.workspace.service_context,
            experiment.name,
            run_id,
            experiment_id=experiment.id
        )
        if name is None:
            name = "run_{}".format(run_id)
        run_dto = client.create_run(
            run_id=run_id,
            target=RunHistoryFacade.target_name(),
            run_name=name,
            properties=properties,
            tags=tags,
            typev2=typev2,
            display_name=display_name,
            description=description
        )
        return run_dto

    @staticmethod
    def create_run_id(run_id=None):
        return run_id if run_id else str(uuid.uuid4())

    @classmethod
    def chain_names(cls, name, child_name):
        name = name if name else ""
        child_name = child_name if child_name else ""
        return "{}{}{}".format(name, RUN_NAME_DELIM, child_name)

    def __enter__(self):
        self._logger.debug("[START]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.upload_tracked_files()

    def patch_run(self, create_run_dto):
        """
        try to update an existing run.
        :param create_run_dto: contains the fields to update
        :type ~_restclient.models.CreateRunDto
        rtype: ~_restclient.models.RunDto
        """
        create_run_dto.run_id = self._run_id
        self.run_dto = self.run.patch_run(create_run_dto)
        return self.run_dto

    def get_run(self):
        self.run_dto = self.run.get_run()
        return self.run_dto

    def refresh_run_dto(self):
        self.run_dto = self.run.get_run()

    def get_runstatus(self):
        run_status = self.run.get_runstatus()
        self.update_cached_status(run_status.status)
        return run_status

    def update_cached_status(self, status):
        self.run_dto.status = status

    def get_status(self):
        self.run_dto = self.run.get_run()
        return self.run_dto.status

    def get_error(self):
        self.run_dto = self.run.get_run()
        return self.run_dto.error

    def get_warnings(self):
        self.run_dto = self.run.get_run()
        return self.run_dto.warnings

    def set_tags(self, tags):
        sanitized_tags = self.run._sanitize_tags(tags)
        create_run_dto = CreateRunDto(run_id=self._run_id, tags=sanitized_tags)
        self.run_dto = self.run.patch_run(create_run_dto)
        return self.run_dto

    def set_display_name(self, display_name):
        create_run_dto = CreateRunDto(run_id=self._run_id, display_name=display_name)
        self.run_dto = self.run.patch_run(create_run_dto)
        return self.run_dto

    def set_description(self, description):
        create_run_dto = CreateRunDto(run_id=self._run_id, description=description)
        self.run_dto = self.run.patch_run(create_run_dto)
        return self.run_dto

    def set_tag(self, key, value):
        return self.set_tags({key: value})

    def get_tags(self):
        self.run_dto = self.run.get_run()
        return self.run_dto.tags

    def delete_tags(self, tags):
        self.run_dto = self.run.delete_run_tags(tags)
        return self.run_dto

    def add_properties(self, properties):
        sanitized_props = self.run._sanitize_tags(properties)
        create_run_dto = CreateRunDto(run_id=self._run_id, properties=sanitized_props)
        self.run_dto = self.run.patch_run(create_run_dto)
        return self.run_dto

    def get_properties(self):
        self.run_dto = self.run.get_run()
        return self.run_dto.properties

    def log_scalar(self, name, value, description="", step=None):
        """Log scalar number as a metric score"""
        metric = ScalarMetric(name, value, description=description, step=step)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_list(self, name, value, description=""):
        """Log list of scalar numbers as a metric score"""
        metric = ListMetric(name, value, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_row(self, name, value, description=""):
        """Log single row of a table as a metric score"""
        metric = RowMetric(name, value, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_table(self, name, value, description=""):
        """Log table as a metric score"""
        metric = TableMetric(name, value, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_confusion_matrix(self, name, value, description=""):
        """Log confusion matrix as a metric score"""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.decoder.JSONDecodeError:
                raise UserErrorException("Invalid JSON provided")
        metric = ConfusionMatrixMetric(name, value, None, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_accuracy_table(self, name, value, description=""):
        """Log accuracy table as a metric score"""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.decoder.JSONDecodeError:
                raise UserErrorException("Invalid JSON provided")
        metric = AccuracyTableMetric(name, value, None, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_residuals(self, name, value, description=""):
        """Log accuracy table as a metric score"""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.decoder.JSONDecodeError:
                raise UserErrorException("Invalid JSON provided")
        metric = ResidualsMetric(name, value, None, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_predictions(self, name, value, description=""):
        """Log accuracy table as a metric score"""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.decoder.JSONDecodeError:
                raise UserErrorException("Invalid JSON provided")
        metric = PredictionsMetric(name, value, None, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric)
        else:
            self._log_metric(metric)

    def log_image(self, name, path=None, plot=None, description=""):
        if path is not None and plot is not None:
            azureml_error = AzureMLError.create(
                TwoInvalidParameter, arg_one="path", arg_two="plot"
            )
            raise AzureMLException._with_error(azureml_error)
        elif path is None and plot is None:
            azureml_error = AzureMLError.create(
                TwoInvalidArgument, arg_one="path", arg_two="plot"
            )
            raise AzureMLException._with_error(azureml_error)
        value = path if path is not None else plot
        metric = ImageMetric(name, value, None, description=description)
        if _use_v2_metrics:
            self._log_metric_v2(metric, is_plot=plot is not None)
        else:
            self._log_metric(metric, is_plot=plot is not None)

    def _log_metric(self, metric, is_plot=False):
        if isinstance(metric, ArtifactBackedMetric):
            # TODO: move this plot stuff up to log_image to avoid passing complexity
            if isinstance(metric, ImageMetric) and is_plot:
                metric.log_to_artifact(self.artifacts, self._origin,
                                       self._data_container_id, is_plot=is_plot)
            else:
                metric.log_to_artifact(self.artifacts, self._origin,
                                       self._data_container_id)
        self.metrics.log(metric)

    def _log_metric_v2(self, metric, is_plot=False):
        if isinstance(metric, ArtifactBackedMetric):
            # TODO: move this plot stuff up to log_image to avoid passing complexity
            if isinstance(metric, ImageMetric) and is_plot:
                metric.log_to_artifact(self.artifacts, self._origin,
                                       self._data_container_id, is_plot=is_plot)
            else:
                metric.log_to_artifact(self.artifacts, self._origin,
                                       self._data_container_id)
        self.metrics.log_v2(metric)

    def get_metrics(self, name=None, recursive=False, run_type=None, populate=False, root_run_id=None, run_ids=None,
                    use_batch=True):
        if recursive and run_ids is not None:
            raise UserErrorException("Cannot recursively get metrics and get metrics for a list of run_ids")

        if recursive:
            # TODO: No better way?
            descendant_ids = [
                child.run_id for child in self.get_descendants(
                    root_run_id=root_run_id,
                    recursive=True,
                    runtype=run_type)
            ]
            run_ids = descendant_ids + [self._run_id]

        if _use_v2_metrics:
            return self.metrics.get_all_metrics_v2(name=name, run_ids=run_ids, populate=populate,
                                                   artifact_client=self.artifacts,
                                                   data_container=self._data_container_id)

        return self.metrics.get_all_metrics(run_ids=run_ids, populate=populate, artifact_client=self.artifacts,
                                            data_container=self._data_container_id, name=name)

    @staticmethod
    def get_runs(experiment, **kwargs):
        """
        :param experiment:
        :type experiment: azureml.core.experiment.Experiment
        :return:
        """
        client = ExperimentClient(experiment.workspace.service_context,
                                  experiment.name,
                                  experiment.id)
        return client.get_runs(**kwargs)

    @staticmethod
    def get_runs_by_compute(workspace, compute_name, **kwargs):
        """
        :param workspace: The workspace object containing the Compute object to retrieve runs from
        :type workspace: azureml.core.Workspace
        :param compute_name:
        :type compute: str
        :return: a generator of ~_restclient.models.RunDto
        """
        client = WorkspaceClient(workspace.service_context)
        return client.get_runs_by_compute(compute_name, **kwargs)

    def get_descendants(self, root_run_id, recursive, **kwargs):
        # Adapter for generator until get_child_runs natively returns a generator of the appropriate
        # subtree
        children = self.run.get_child_runs(root_run_id, recursive=recursive, **kwargs)
        for child in children:
            yield child

    def register_model(self, model_name, model_path=None, tags=None, properties=None,
                       model_framework=None, model_framework_version=None, asset_id=None, sample_input_dataset=None,
                       sample_output_dataset=None, resource_configuration=None, **kwargs):
        """
        Register a model with AML
        :param model_name: model name
        :type model_name: str
        :param model_path: relative cloud path to model from outputs/ dir. When model_path is None, model_name is path.
        :type model_path: str
        :param tags: Dictionary of key value tags to give the model
        :type tags: dict[str, str]
        :param properties: Dictionary of key value properties to give the model. These properties cannot
            be changed after model creation, however new key value pairs can be added
        :type properties: dict[str, str]
        :param model_framework: Framework of the model to register.
            Currently supported frameworks: TensorFlow, ScikitLearn, Onnx, Custom, Multi
        :type model_framework: str
        :param model_framework_version: The framework version of the registered model
        :type model_framework_version: str
        :param asset_id: id of existing asset
        :type asset_id: str
        :param sample_input_dataset: Optional, Sample input dataset for the registered model
        :type sample_input_dataset: TabularDataset | FileDataset
        :param sample_output_dataset: Optional, Sample output dataset for the registered model
        :type sample_output_dataset: TabularDataset | FileDataset
        :param resource_configuration: Optional, Resource configuration to run the registered model
        :type resource_configuration: azureml.core.resource_configuration.ResourceConfiguration
        :rtype: azureml.core.model.Model
        """
        if model_path is None:
            model_path = model_name
        model_path = os.path.normpath(model_path)
        model_path = model_path.replace(os.sep, '/')

        if self._data_container_id is None:
            raise UserErrorException("Data Container ID cannot be null for run with ID {0}".format(self._run_id))

        artifact_prefix_id = "ExperimentRun/{}/{}".format(self._data_container_id, model_path)
        cloud_file_paths = list(self.artifacts.get_files_by_artifact_prefix_id(artifact_prefix_id))
        if not cloud_file_paths:
            run_files = list(self.artifacts.get_file_paths(self._origin, self._data_container_id))
            raise ModelPathNotFoundException(
                """Could not locate the provided model_path {} in the set of files uploaded to the run: {}
                See https://aka.ms/run-logging for more details.""".format(model_path, str(run_files)))
        artifacts = [{"prefix": artifact_prefix_id}]
        metadata_dict = None
        if asset_id is None:
            asset = self.assets.create_asset(model_name,
                                             artifacts,
                                             metadata_dict=metadata_dict,
                                             run_id=self._run_id)
            asset_id = asset.id
        else:
            # merge asset tags and properties with those from model
            asset = self.assets.get_asset_by_id(asset_id)
            properties = self._merge_dict(tags, asset.tags)
            properties = self._merge_dict(properties, asset.properties)

        model = Model._register_with_asset(self._experiment.workspace, model_name, asset_id, tags=tags,
                                           properties=properties, experiment_name=self._experiment.name,
                                           run_id=self._run_id, model_framework=model_framework,
                                           model_framework_version=model_framework_version,
                                           sample_input_dataset=sample_input_dataset,
                                           sample_output_dataset=sample_output_dataset,
                                           resource_configuration=resource_configuration, **kwargs)
        return model

    @staticmethod
    def _merge_dict(dict_aa, dict_bb):
        """
            Returns merged dict that contains dict_aa and any item in dict_bb not in dict_aa
        :param dict_aa:
        :param dict_bb:
        """
        if dict_aa is None:
            return dict_bb.copy()
        elif dict_bb is None:
            return dict_aa.copy()
        else:
            result = dict_aa.copy()
            result.update(dict_bb)
            return result

    @staticmethod
    def _error_details_to_dictionary(error_details=None, error_code=None):
        """
            Returns the correct dictionary of values to log an error event to a run
        :param error_details: Instance of string or BaseException
        :param error_code: Optional error code of the error for the error classification
        """
        error_dict = {"code": "UserError" if error_code is None else error_code}
        if error_details is None:
            return error_dict

        if isinstance(error_details, six.text_type):
            error_dict["message"] = error_details
        elif isinstance(error_details, BaseException):
            tb = error_details.__traceback__
            exception_type = type(error_details).__name__
            stack_trace = ''.join(traceback.format_tb(tb))

            if isinstance(error_details, ServiceException) and hasattr(error_details, "error"):
                # Service exceptions will contain an ErrorResponse already formatted with error codes, messages,
                # target etc. Extract that dictionary of fields (called RootError)
                error_response = json.loads(str(error_details.error))
                error_dict = error_response["error"]  # RootError
            elif isinstance(error_details, AzureMLException):
                error_dict = json.loads(error_details._serialize_json())['error']
            else:
                error_dict["message"] = \
                    "User program failed with {}: {}".format(exception_type, str(error_details))

            error_dict["debug_stack_trace"] = stack_trace
            error_dict["debug_type"] = exception_type
            error_dict["debug_message"] = str(error_details)

            if error_dict.get("details_uri") is None:
                error_dict["details_uri"] = "https://aka.ms/azureml-run-troubleshooting"
        else:
            raise TypeError("error_details must be instance of string or BaseException.")

        return error_dict

    def create_child_run(self, name, target, child_name=None, run_id=None, properties=None, tags=None):
        """
        Creates a child run
        :param name: Name of the current run
        :type name: str:
        :param child_name: Optional name to set for the child run object
        :type child_name: str:
        :param run_id: Optional run_id to set for run, otherwise defaults
        :type run_id: str:
        :param properties: Optional initial properties of a run
        :type properties: dict[str]
        :param tags: Optional initial tags on a run
        :type tags: dict[str]
        """
        sanitized_tags = self.run._sanitize_tags(tags) if tags else None
        sanitized_properties = self.run._sanitize_tags(properties) if properties else None
        child_run_id = run_id if run_id else RunHistoryFacade.create_run_id(run_id)
        child_name = RunHistoryFacade.chain_names(name, child_name)
        child = self.run.create_child_run(
            child_run_id, target=target, run_name=child_name, properties=sanitized_properties, tags=sanitized_tags)
        return child

    def create_children(self, tag_key, tag_values, start_children=True):
        """
        Creates one child for each element in tag_values
        :param tag_key: key for the Tags entry to populate in all created children
        :type tag_key: str:
        :param tag_Values: list of values that will map onto Tags[tag_key] for the list of runs created
        :type tag_values: [str]
        :param start_children: Optional flag to start created children, defaults True
        :type start_children: bool:
        :rtype [RunDto]
        """
        request_child_runs = []
        for tag_value in tag_values:
            create_run_dto = CreateRunDto(run_id=RunHistoryFacade.create_run_id(),
                                          parent_run_id=self._run_id,
                                          status='NotStarted',
                                          tags={tag_key: tag_value})
            request_child_runs.append(create_run_dto)
        result_dto = self.run.batch_create_child_runs(request_child_runs)
        errors = result_dto.errors
        if len(errors) > 0:
            azureml_error = AzureMLError.create(
                CreateChildrenFailed, run_id='runid'
            )
            raise AzureMLException._with_error(azureml_error)
        result_child_runs = result_dto.runs
        child_run_ids = [child_run.run_id for child_run in request_child_runs]
        if start_children:
            event_errors = self.run.batch_post_event_start(child_run_ids).errors
            if len(event_errors) > 0:
                azureml_error = AzureMLError.create(
                    StartChildrenFailed, run_id='runid'
                )
                raise AzureMLException._with_error(azureml_error)
        return (result_child_runs[run_id] for run_id in child_run_ids)

    def start(self):
        """
        Changes the state of the current run to started
        """
        self.run.post_event_start(caller=self.identity)

    def complete(self, _set_status=True):
        """
        Changes the state of the current run to completed
        """
        self.flush()
        self.upload_tracked_files()
        self.set_completed_status(_set_status=_set_status)

    def set_completed_status(self, _set_status):
        if _set_status:
            self.run.post_event_completed(caller=self.identity)

    def fail(self, error_details=None, error_code=None, _set_status=True):
        """
        Changes the state of the current run to failed

        Optionally set the Error property of the run with a message or exception passed to error_details.

        :param name: error_details
        :type name: Instance of string or BaseException
        :param error_code: Optional error code of the error for the error classification
        :type error_code: str
        """
        self.flush()
        self.upload_tracked_files()
        self.set_failed_status(_set_status=_set_status, error_details=error_details, error_code=error_code)

    def set_failed_status(self, _set_status, error_details=None, error_code=None):
        error_dict = self._error_details_to_dictionary(error_details=error_details, error_code=error_code)
        self.run.post_event_error(error_dict, caller=self.identity)

        if _set_status:
            self.run.post_event_failed(caller=self.identity)

    def cancel(self, uri=None):
        """
        Changes the state of the current run to canceled
        """
        if uri:
            auth = self.run._service_context.get_auth()
            headers = auth.get_authentication_header()
            with create_session_with_retry() as session:
                ClientBase._execute_func(session.post, uri, headers=headers)
        else:
            self.run.post_event_canceled()
        self.flush()
        self.upload_tracked_files()

    def diagnostics(self, uri):
        """
        Retrieves the diagnostics in the working directory of the current run.
        """
        auth = self.run._service_context.get_auth()
        headers = auth.get_authentication_header()
        with create_session_with_retry() as session:
            try:
                response = ClientBase._execute_func(session.get, uri, headers=headers)
                response.raise_for_status()
            except requests.exceptions.HTTPError:
                raise WebserviceException('Received bad response from Execution Service:\n'
                                          'Response Code: {}\n'
                                          'Headers: {}\n'
                                          'Content: {}'.format(response.status_code, response.headers,
                                                               response.content),
                                          logger=module_logger)
        return response

    def flush(self, timeout_seconds=300):
        self.metrics.flush(timeout_seconds=timeout_seconds)

    def upload_tracked_files(self):
        self.output_file_tracker.upload_tracked_files()

    def take_snapshot(self, file_or_folder_path, _raise_on_validation_failure=True):
        """Save a snapshot of the inputted file or folder.

        :param file_or_folder_path: The file or folder containing the run source code.
        :type file_or_folder_path: str
        :return: Returns the snapshot id
        :param _raise_on_validation_failure: If set to True (by default), will raise an exception on validation errors
        :type _raise_on_validation_failure: bool
        :rtype: str
        """
        # Ensure we don't already have one
        existing_snapshot = self.get_snapshot_id()
        if existing_snapshot is not None:
            raise SnapshotException("Cannot take snapshot as the run already has one: {}".format(existing_snapshot))

        with self._log_context("TakingSnapshot") as slcx:
            snapshot_id = self.snapshots.create_snapshot(
                file_or_folder_path,
                raise_on_validation_failure=_raise_on_validation_failure)

            slcx.debug("Created snapshot {}".format(snapshot_id))
            self.add_properties({
                PropertyKeys.SNAPSHOT_ID: snapshot_id
            })
        return snapshot_id

    def get_snapshot_id(self):
        properties = self.get_properties()
        return properties.get(PropertyKeys.SNAPSHOT_ID, None)

    def _update_dataset_lineage(self, datasets):
        from azureml.data._dataset import _Dataset
        from azureml._restclient.models import DatasetLineage, DatasetIdentifier, DatasetConsumptionType

        if not datasets or len(datasets) == 0:
            return

        new_datasets = []
        for dataset in datasets:
            if isinstance(dataset, _Dataset):
                if not dataset.id:
                    continue
                new_datasets.append(DatasetLineage(
                    identifier=DatasetIdentifier(saved_id=dataset.id),
                    consumption_type=DatasetConsumptionType.reference
                ))

        create_run_dto = CreateRunDto(
            run_id=self._run_id,
            input_datasets=new_datasets
        )
        self.patch_run(create_run_dto)

    def _update_output_dataset_lineage(self, output_datasets):
        create_run_dto = CreateRunDto(run_id=self._run_id, output_datasets=output_datasets)
        self.patch_run(create_run_dto)
