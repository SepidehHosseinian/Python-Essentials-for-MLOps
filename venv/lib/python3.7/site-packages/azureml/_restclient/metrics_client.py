# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Access metrics client"""
import logging
import os
import time

from collections import defaultdict

from azureml.core._metrics import (Artifact, InlineMetric, INLINE_METRICS,
                                   AZUREML_TABLE_METRIC_TYPE, AZUREML_OLD_IMAGE_METRIC_TYPE)

from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import MetricsNumberExceeds
from azureml.exceptions import AzureMLException, ServiceException, UserErrorException
from azureml._logging import START_MSG, STOP_MSG
from azureml._async import TaskQueue, BatchTaskQueue

from ._odata.runs import get_filter_expression
from .run_client import RunClient
from .models.batch_metric_dto import BatchMetricDto
from .models.batch_metric_v2_dto import BatchMetricV2Dto
from .models.retrieve_full_fidelity_metric_request_dto import RetrieveFullFidelityMetricRequestDto
from .models.list_metrics_dto import ListMetricsDto
from .models.post_run_metrics_result_dto import PostRunMetricsResultDto
from .utils import _generate_client_kwargs

module_logger = logging.getLogger(__name__)

AZUREML_METRICS_CLIENT_BATCH_SIZE_ENV_VAR = "AZUREML_METRICS_BATCH_SIZE"
AZUREML_METRICS_CLIENT_BATCH_CUSHION_ENV_VAR = "AZUREML_METRICS_BATCH_CUSHION"
AZUREML_METRICS_CLIENT_POLLING_INTERVAL_ENV_VAR = "AZUREML_METRICS_POLLING_INTERVAL"

AZUREML_MAX_NUMBER_METRICS_BATCH = 50
AZUREML_METRICS_CLIENT_BATCH_CUSHION = 5
AZUREML_METRICS_CLIENT_MIN_POLLING_INTERVAL = 1  # Minimum supported interval

_use_v2_metrics = os.getenv('AZUREML_USE_METRICS_V2', True)


class MetricsClient(RunClient):
    """Metrics client class"""

    def __init__(self, *args, **kwargs):
        self._use_batch = kwargs.pop("use_batch", True)
        self._flush_eager = kwargs.pop("flush_eager", False)
        super(MetricsClient, self).__init__(*args, **kwargs)
        self._task_queue_internal = None
        self._task_queue_v2_internal = None
        self._metricid_values_dict = {}

    @property
    def _task_queue(self):
        if self._task_queue_internal is not None:
            return self._task_queue_internal

        if self._use_batch:
            max_batch_size = int(os.environ.get(AZUREML_METRICS_CLIENT_BATCH_SIZE_ENV_VAR,
                                                AZUREML_MAX_NUMBER_METRICS_BATCH))
            batch_cushion = int(os.environ.get(AZUREML_METRICS_CLIENT_BATCH_CUSHION_ENV_VAR,
                                               AZUREML_METRICS_CLIENT_BATCH_CUSHION))
            interval = int(os.environ.get(AZUREML_METRICS_CLIENT_POLLING_INTERVAL_ENV_VAR,
                                          AZUREML_METRICS_CLIENT_MIN_POLLING_INTERVAL))
            self._logger.debug(
                "Overrides: Max batch size: {0}, batch cushion: {1}, Interval: {2}.".format(
                    max_batch_size, batch_cushion, interval))

            self._task_queue_internal = BatchTaskQueue(self._log_batch,
                                                       worker_pool=self._pool,
                                                       interval=interval,
                                                       max_batch_size=max_batch_size,
                                                       batch_cushion=batch_cushion,
                                                       _ident="PostMetricsBatch",
                                                       _parent_logger=self._logger)
        else:
            self._task_queue_internal = TaskQueue(worker_pool=self._pool,
                                                  _ident="PostMetricsSingle",
                                                  _parent_logger=self._logger)

        self._logger.debug("Used {} for use_batch={}.".format(type(self._task_queue_internal), self._use_batch))
        return self._task_queue_internal

    @property
    def _task_queue_v2(self):
        if self._task_queue_v2_internal is not None:
            return self._task_queue_v2_internal
        if self._use_batch:
            max_batch_size = int(os.environ.get(AZUREML_METRICS_CLIENT_BATCH_SIZE_ENV_VAR,
                                                AZUREML_MAX_NUMBER_METRICS_BATCH))
            batch_cushion = int(os.environ.get(AZUREML_METRICS_CLIENT_BATCH_CUSHION_ENV_VAR,
                                               AZUREML_METRICS_CLIENT_BATCH_CUSHION))
            interval = int(os.environ.get(AZUREML_METRICS_CLIENT_POLLING_INTERVAL_ENV_VAR,
                                          AZUREML_METRICS_CLIENT_MIN_POLLING_INTERVAL))
            self._logger.debug(
                "Overrides: Max batch size: {0}, batch cushion: {1}, Interval: {2}.".format(
                    max_batch_size, batch_cushion, interval))
            self._task_queue_v2_internal = BatchTaskQueue(self._log_batch_v2,
                                                          worker_pool=self._pool,
                                                          interval=interval,
                                                          max_batch_size=max_batch_size,
                                                          batch_cushion=batch_cushion,
                                                          _ident="PostMetricsBatchV2",
                                                          _parent_logger=self._logger)
        else:
            self._task_queue_v2_internal = TaskQueue(worker_pool=self._pool, _ident="PostMetricsSingleV2",
                                                     _parent_logger=self._logger)

        self._logger.debug("Used {} for use_batch={}.".format(type(self._task_queue_v2_internal), self._use_batch))
        return self._task_queue_v2_internal

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_metrics_restclient(user_agent=user_agent)

    @classmethod
    def dto_v2_to_metric_cells(cls, metric_dto, artifact_client=None, populate=False):
        metric_type = metric_dto.properties.ux_metric_type
        if metric_type in INLINE_METRICS:
            return InlineMetric.get_cells_from_metric_v2_dto(metric_dto)

        if populate and artifact_client is None:
            raise UserErrorException('Cannot use populate with no artifact_client provided')

        # Artifact backed metrics are defined by the first value uploaded whether there are one or many
        if len(metric_dto.value) >= 1 and metric_dto.name in metric_dto.columns:
            data_location = metric_dto.value[0].data[metric_dto.name]
            artifact = Artifact.create_unpopulated_artifact_from_data_location(data_location,
                                                                               metric_type,
                                                                               metric_dto.data_container_id)
            if populate:
                artifact.retrieve_artifact(artifact_client)
            return artifact if populate else artifact.data_location

    @classmethod
    def dto_to_metrics_dict(cls, metrics_dto, artifact_client=None, populate=False,
                            data_container=None):
        """convert metrics_dto to dictionary"""
        out = {}
        metric_types = {}
        original_cell_types = {}
        seen_metric_ids = set({})
        for metric in metrics_dto:
            metric_id = metric.metric_id
            if metric_id in seen_metric_ids:
                continue
            name = metric.name
            data_location = metric.data_location
            metric_type = metric.metric_type
            schema = metric.schema
            cell_types = {prop.property_id: prop.type for prop in schema.properties}
            if name not in metric_types:
                metric_types[name] = metric_type
                original_cell_types[name] = cell_types

            if metric_types[name] == metric_type:
                if metric_type in INLINE_METRICS:
                    if metric_type == AZUREML_TABLE_METRIC_TYPE:
                        InlineMetric.add_table(name, out, original_cell_types[name], metric.cells, cell_types)
                    else:
                        InlineMetric.add_cells(name, out, original_cell_types[name], metric.cells, cell_types)
                else:
                    if metric_type == AZUREML_OLD_IMAGE_METRIC_TYPE:
                        data_location = metric.cells[0][name]

                    artifact = Artifact.create_unpopulated_artifact_from_data_location(data_location,
                                                                                       metric_type,
                                                                                       data_container)
                    if populate:
                        artifact.retrieve_artifact(artifact_client)
                    value = artifact if populate else artifact.data_location

                    if name in out:
                        if not isinstance(out[name], list):
                            out[name] = [out[name]]
                        out[name].append(value)
                    else:
                        out[name] = value
            else:
                module_logger.warning("Conflicting metric types, logged metric "
                                      "of type: {} to name: {}, expected metric "
                                      "type is {}".format(metric_type, name,
                                                          metric_types[name]))
            seen_metric_ids.add(metric_id)

        return out

    def _get_metrics_for_run(self, metric_types=None, after_timestamp=None, name=None,
                             merge_strategy_type=None, custom_headers=None):
        """get metrics with the same run_id"""
        expression = get_filter_expression(run_ids=[self._run_id],
                                           metric_types=metric_types,
                                           metric_name=name,
                                           after_timestamp=after_timestamp)

        kwargs = _generate_client_kwargs(has_query_params=True, filter=expression, custom_headers=custom_headers)

        return self._execute_with_experiment_arguments(self._client.run_metric.get_by_query,
                                                       mergestrategytype=merge_strategy_type,
                                                       is_paginated=True,
                                                       **kwargs)

    def _get_metrics_for_runs(self, run_ids, use_batch=True, metric_types=None, after_timestamp=None, name=None,
                              merge_strategy_type=None, custom_headers=None):
        metric_dtos_by_run = defaultdict(list)

        if use_batch:
            tasks = self.get_metrics_in_batches_by_run_ids(run_ids,
                                                           metric_types=metric_types,
                                                           after_timestamp=after_timestamp,
                                                           name=name,
                                                           merge_strategy_type=merge_strategy_type,
                                                           custom_headers=custom_headers)
        else:
            tasks = [self.get_metrics_by_run_ids(run_ids=run_ids,
                                                 metric_types=metric_types,
                                                 after_timestamp=after_timestamp,
                                                 name=name,
                                                 merge_strategy_type=merge_strategy_type,
                                                 custom_headers=custom_headers)]

        for task in tasks:
            for dto in task:
                metric_dtos_by_run[dto.run_id].append(dto)

        return metric_dtos_by_run

    @staticmethod
    def convert_metrics_to_objects(metrics):
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, Artifact):
                metrics[metric_name] = metric_value.convert_to_object()
        return metrics

    def get_metric_dto_for_run_v2(self, name, run_id=None, start_time=None, end_time=None,
                                  custom_headers=None):
        request_dto = RetrieveFullFidelityMetricRequestDto(metric_name=name,
                                                           start_time=start_time,
                                                           end_time=end_time)
        kwargs = _generate_client_kwargs(custom_headers=custom_headers)
        if run_id is not None:
            kwargs["run_id"] = run_id

        try:
            metric_dto = self._execute_with_workspace_run_arguments(
                self._client.metric.get_full_fidelity, request_dto, **kwargs)
        except ServiceException as e:
            # v2 GetFullFidelity returns a 404 when metric does not exist unlike the v1 QueryRunMetric endpoint
            if e.status_code == 404:
                return None
            raise
        continuation_token = metric_dto.continuation_token
        while continuation_token is not None:
            request_dto.continuation_token = continuation_token
            temp_metric_dto = self._execute_with_workspace_run_arguments(self._client.metric.get_full_fidelity,
                                                                         request_dto, **kwargs)
            continuation_token = temp_metric_dto.continuation_token
            metric_dto.value += temp_metric_dto.value
        return metric_dto

    def get_metric_for_run_v2(self, name, run_id=None, start_time=None, end_time=None, artifact_client=None,
                              populate=False, custom_headers=None):
        run_id = self._run_id if run_id is None else run_id
        metric_dto = self.get_metric_dto_for_run_v2(name, run_id=run_id, start_time=start_time, end_time=end_time,
                                                    custom_headers=custom_headers)
        if metric_dto is None or len(metric_dto.value) == 0:
            return None
        return MetricsClient.dto_v2_to_metric_cells(metric_dto, artifact_client, populate)

    def get_metrics_for_run_v2(self, run_id=None, name=None, start_time=None, end_time=None, populate=False,
                               artifact_client=None, data_container=None, custom_headers=None):
        returned_metrics = {}
        run_id = self._run_id if run_id is None else run_id
        if name is not None:
            metric = self.get_metric_for_run_v2(name, run_id=run_id, start_time=start_time, end_time=end_time,
                                                artifact_client=artifact_client, populate=populate,
                                                custom_headers=custom_headers)
            if metric is not None:
                returned_metrics[name] = metric
        else:
            try:
                metric_names = self.list_metric_names_for_run_id(run_id)
            except Exception:
                self._logger.debug("Error getting metrics for run id {}".format(run_id))
                return returned_metrics
            for metric_name in metric_names:
                metric = self.get_metric_for_run_v2(metric_name, run_id=run_id, start_time=start_time,
                                                    end_time=end_time, artifact_client=artifact_client,
                                                    populate=populate, custom_headers=custom_headers)
                if metric is not None:
                    returned_metrics[metric_name] = metric
        return returned_metrics

    def get_metrics_for_run_ids_v2(self, name=None, run_ids=None, start_time=None, end_time=None,
                                   populate=False, artifact_client=None, data_container=None,
                                   custom_headers=None):
        returned_metrics_for_runs = {}
        all_run_ids = [self._run_id] if run_ids is None else run_ids
        for run_id in all_run_ids:
            metrics = self.get_metrics_for_run_v2(run_id=run_id, name=name, start_time=start_time,
                                                  end_time=end_time, populate=populate,
                                                  artifact_client=artifact_client, data_container=data_container,
                                                  custom_headers=custom_headers)
            if len(metrics) != 0:
                returned_metrics_for_runs[run_id] = metrics
        if run_ids is None and self._run_id in returned_metrics_for_runs:
            return returned_metrics_for_runs[self._run_id]
        return returned_metrics_for_runs

    def get_all_metrics_v2(self, name=None, run_ids=None, populate=False, artifact_client=None,
                           data_container=None, after_timestamp=None, custom_headers=None):
        # Compared to v1, the following are ignored; convert_to_object (default False), use_batch(default True),
        # metric_types and merge_strategy_type, all are not exposed to the SDK users.
        return self.get_metrics_for_run_ids_v2(name=name, run_ids=run_ids, start_time=after_timestamp,
                                               populate=populate, artifact_client=artifact_client,
                                               data_container=data_container, custom_headers=custom_headers)

    def get_all_metrics(self, run_ids=None, populate=False, artifact_client=None,
                        convert_to_object=False, use_batch=True, metric_types=None,
                        data_container=None, after_timestamp=None, name=None,
                        merge_strategy_type=None, custom_headers=None):

        metric_dtos_by_run = self._get_all_metrics(run_ids=run_ids, use_batch=use_batch,
                                                   metric_types=metric_types,
                                                   name=name,
                                                   after_timestamp=after_timestamp,
                                                   merge_strategy_type=merge_strategy_type,
                                                   custom_headers=custom_headers)
        if convert_to_object:
            metrics = {
                runid: MetricsClient.convert_metrics_to_objects(
                    self.dto_to_metrics_dict(metric_dto_list, populate=populate,
                                             artifact_client=artifact_client,
                                             data_container=data_container)
                ) for runid, metric_dto_list in metric_dtos_by_run.items()
            }
        else:
            metrics = {
                runid: self.dto_to_metrics_dict(metric_dto_list, populate=populate,
                                                artifact_client=artifact_client,
                                                data_container=data_container)
                for runid, metric_dto_list in metric_dtos_by_run.items()
            }

        if run_ids is None:
            return metrics[self._run_id]
        return metrics

    def _get_all_metrics(self, run_ids=None, use_batch=True, metric_types=None, after_timestamp=None,
                         name=None, merge_strategy_type=None, custom_headers=None):
        if run_ids is None:
            metric_dtos_by_run = {self._run_id: self._get_metrics_for_run(metric_types=metric_types, name=name,
                                                                          after_timestamp=after_timestamp,
                                                                          merge_strategy_type=merge_strategy_type,
                                                                          custom_headers=custom_headers)}
        else:
            metric_dtos_by_run = self._get_metrics_for_runs(run_ids,
                                                            use_batch=use_batch,
                                                            metric_types=metric_types,
                                                            after_timestamp=after_timestamp,
                                                            name=name,
                                                            merge_strategy_type=merge_strategy_type,
                                                            custom_headers=custom_headers)

        return metric_dtos_by_run

    def list_metric_names_for_run_id(self, run_id=None, custom_headers=None):
        request_dto = ListMetricsDto()
        kwargs = _generate_client_kwargs(custom_headers=custom_headers)
        if run_id is not None:
            kwargs["run_id"] = run_id
        metric_def_dtos = self._execute_with_workspace_run_arguments(self._client.metric.list_metrics,
                                                                     request_dto,
                                                                     is_paginated=True,
                                                                     **kwargs)
        result = []
        for metric_def_dto in metric_def_dtos:
            result.append(metric_def_dto.metric_key.name)
        return result

    def _metric_conversion_for_widgets(self, metrics):
        # convert non-(list, dict) metrics (i.e. scalar metrics logged only once to lists)
        # this is necessary because current widget code expects everything as a list or dict
        for run_id, run_metrics in metrics.items():
            for metric_name, metric_value in run_metrics.items():
                if isinstance(metric_value, list) or isinstance(metric_value, dict):
                    continue
                else:
                    metrics[run_id][metric_name] = [metric_value]
        return metrics

    def flush(self, timeout_seconds=120):
        with self._log_context("FlushingMetricsClient"):
            self._task_queue.flush(self.identity, timeout_seconds=timeout_seconds)
            self._task_queue_v2.flush(self.identity, timeout_seconds=timeout_seconds)

        if os.getenv('AZUREML_FLUSH_INGEST_WAIT', True):
            metrics_ingest_timeout_seconds = int(os.getenv('AZUREML_FLUSH_INGEST_WAIT_TIMEOUT_SECONDS', 240))
            self.wait_for_metrics_ingest(timeout_seconds=metrics_ingest_timeout_seconds)

    def log(self, metric, custom_headers=None):
        metric_dto = metric.create_metric_dto()
        if isinstance(self._task_queue, BatchTaskQueue):
            self._task_queue.add_item(metric_dto)
        else:
            ident = "{}_{}".format(self._task_queue._tasks.qsize(), self._task_queue._identity)
            kwargs = _generate_client_kwargs(custom_headers=custom_headers)
            async_task = self._execute_with_run_arguments(self._client.run_metric.post,
                                                          metric_dto,
                                                          is_async=True,
                                                          new_ident=ident,
                                                          **kwargs)
            self._task_queue.add_task(async_task)
            if not self._flush_eager:
                return async_task

        if self._flush_eager:
            self.flush()

    def log_v2(self, metric, custom_headers=None):
        metric_v2_dto = metric.create_v2_dto()
        for val in metric_v2_dto.value:
            metricId = val.metric_id
            value = list(val.data.values())
            if self._metricid_values_dict.get(metricId) is None:
                self._metricid_values_dict[metricId] = value
            else:
                self._logger.debug("Regenerating same metricId {} for this run".format(
                    self._metricid_values_dict.get(metricId)))

        if metric_v2_dto is not None:
            if isinstance(self._task_queue_v2, BatchTaskQueue):
                self._task_queue_v2.add_item(metric_v2_dto)
            else:
                kwargs = _generate_client_kwargs(custom_headers=custom_headers)
                batch_metric_v2_dto = BatchMetricV2Dto(values=[metric_v2_dto], report_errors=True)
                self._logger.debug("Metrics Client: logs_v2 is calling post_run_metrics with "
                                   "{} values.".format(len(batch_metric_v2_dto.values)))
                async_task = self._execute_with_workspace_run_arguments(self._post_run_metrics_log_failed_validations,
                                                                        batch_metric_v2_dto,
                                                                        is_async=True,
                                                                        **kwargs)
                self._task_queue_v2.add_task(async_task)
                if not self._flush_eager:
                    return async_task

            if self._flush_eager:
                self.flush()

    def _log_batch(self, metric_dtos, is_async=False):
        if len(metric_dtos) > AZUREML_MAX_NUMBER_METRICS_BATCH:
            azureml_error = AzureMLError.create(
                MetricsNumberExceeds, metric_dtos=len(metric_dtos),
                AZUREML_MAX_NUMBER_METRICS_BATCH=AZUREML_MAX_NUMBER_METRICS_BATCH
            )
            raise AzureMLException._with_error(azureml_error)

        batch_metric_dto = BatchMetricDto(metric_dtos)
        res = self._execute_with_run_arguments(self._client.run_metric.post_batch,
                                               batch_metric_dto,
                                               is_async=is_async)
        return res

    def _log_batch_v2(self, metric_dtos, is_async=False):
        if len(metric_dtos) > AZUREML_MAX_NUMBER_METRICS_BATCH:
            azureml_error = AzureMLError.create(
                MetricsNumberExceeds, metric_dtos=len(metric_dtos),
                AZUREML_MAX_NUMBER_METRICS_BATCH=AZUREML_MAX_NUMBER_METRICS_BATCH
            )
            raise AzureMLException._with_error(azureml_error)

        batch_metric_dto = BatchMetricV2Dto(values=metric_dtos, report_errors=True)
        self._logger.debug("Metrics Client: _log_batch_v2 is calling post_run_metrics "
                           "posting {} values.".format(len(batch_metric_dto.values)))
        try:
            res = self._execute_with_workspace_run_arguments(self._post_run_metrics_log_failed_validations,
                                                             batch_metric_dto, is_async=is_async)
        except BaseException as e:

            # This is to retry this operation as os.environment is changing due to its usage in multiple thread
            # because of which we are getting error related to "dictionary changed size during iteration"
            # More details is available in the work item :-
            # https://msdata.visualstudio.com/Vienna/_workitems/edit/1064675
            self._logger.warning("Metrics Client: _execute_with_workspace_run_arguments failed , \
                               retrying it after 3 seconds. Reason for failure {0}".format(str(e)))
            time.sleep(3)
            res = self._execute_with_workspace_run_arguments(self._post_run_metrics_log_failed_validations,
                                                             batch_metric_dto, is_async=is_async)
        return res

    def _post_run_metrics_log_failed_validations(self, subscription_id, resource_group_name, workspace_name,
                                                 run_id, batch_metric_v2_dto, custom_headers=None, raw=False,
                                                 **operation_config):
        res = self._client.metric.post_run_metrics(subscription_id, resource_group_name, workspace_name,
                                                   run_id, batch_metric_v2_dto, custom_headers=custom_headers, raw=raw,
                                                   **operation_config)
        if isinstance(res, PostRunMetricsResultDto):
            if len(res.errors) >= 1:
                for error in res.errors:
                    message = error.error_response.error.message
                    self._logger.warning("Metrics Client: Failed to post metric with name {} "
                                         "due to the following error \"{}\"".format(error.metric.name, message))

    def __enter__(self):
        self._logger.debug(START_MSG)
        return self

    def __exit__(self, *args):
        self.flush()
        self._logger.debug(STOP_MSG)
