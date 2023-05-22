# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Access a experiment client"""
import copy
import six

from azureml._base_sdk_common import _ClientSessionId

from .contracts.utils import get_new_id
from .constants import DEFAULT_PAGE_SIZE
from .models.create_run_dto import CreateRunDto
from .utils import _generate_client_kwargs, _validate_order_by
from .workspace_client import WorkspaceClient
from ._odata.constants import ORDER_BY_STARTTIME_EXPRESSION, ORDER_BY_RUNID_EXPRESSION
from ._odata.runs import (get_run_ids_filter_expression, get_filter_expression)


class ExperimentClient(WorkspaceClient):
    """
    Experiment APIs

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
    :param experiment_name:
    :type experiment_name: str
    """

    def __init__(self,
                 service_context,
                 experiment_name,
                 experiment_id=None,
                 **kwargs):
        """
        Constructor of the class.
        """
        super(ExperimentClient, self).__init__(service_context, **kwargs)
        self._experiment_name = experiment_name
        self._experiment_id = experiment_id
        self._experiment_arguments = copy.deepcopy(self._workspace_arguments)
        self._experiment_arguments.append(self._experiment_name)
        self._experiment_arguments_with_experiment_id = copy.deepcopy(self._workspace_arguments)
        self._experiment_arguments_with_experiment_id.append(self._experiment_id)

    def create_run(self, run_id=None, script_name=None, target=None, run_name=None, create_run_dto=None,
                   properties=None, tags=None, caller=None, custom_headers=None, is_async=False, typev2=None,
                   display_name=None, description=None):
        """
        create a run
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param run_id: run id
        :type run_id: str
        :param script_name: script name
        :type script_name: str
        :param target: run target
        :type target: str
        :param run_name: run name
        :type run_name: str
        :param CreateRunDto create_run_dto: run object to create
        :type create_run_dto: CreateRunDto
        :param properties: Initial set of properties on the run
        :type properties: dict
        :param tags: Initial set of tags on the run
        :type tags: dict
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_header: dict
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.RunDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        run_id = get_new_id() if run_id is None else run_id
        if not create_run_dto or not isinstance(create_run_dto, CreateRunDto):
            create_run_dto = CreateRunDto(run_id=run_id,
                                          script_name=script_name,
                                          target=target,
                                          name=run_name,
                                          properties=properties,
                                          tags=tags,
                                          run_type_v2=typev2,
                                          display_name=display_name, description=description)

        kwargs = _generate_client_kwargs(create_run_dto=create_run_dto,
                                         is_async=is_async, caller=caller, custom_headers=custom_headers)

        kwargs['run_id'] = run_id
        return self._execute_with_experiment_arguments(self._client.run.patch, **kwargs)

    def get_run(self, run_id, caller=None, custom_headers=None, is_async=False):
        """
        Get detail of a run by its run_id
        :param run_id: run id (required)
        :type run_id: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.RunDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            is_async=is_async, caller=caller, custom_headers=custom_headers)

        if self._experiment_id:
            return self._execute_with_experimentid_arguments(self._client.run.get_by_exp_id,
                                                             run_id=run_id,
                                                             **kwargs)
        return self._execute_with_experiment_arguments(self._client.run.get,
                                                       run_id=run_id,
                                                       **kwargs)

    def get_runs(self, last=0, _filter_on_server=True,
                 page_size=DEFAULT_PAGE_SIZE, order_by=None,
                 caller=None, custom_headers=None, **kwargs):
        """
        Get detail of all runs of an experiment
        :param last: the number of latest runs to return (optional)
        :type last: int
        :param page_size: number of dto returned by one request (optional)
        :type page_size: int
        :param order_by: keys to sort return values, ('sort_key', 'asc'/'desc')(optional)
        :type order_by: tuple (str, str)
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return: a generator of ~_restclient.models.RunDto
        """
        order_by_expression = _validate_order_by(order_by) if order_by else [ORDER_BY_STARTTIME_EXPRESSION]
        pagesize = DEFAULT_PAGE_SIZE if page_size < 1 else page_size
        top = last if last < pagesize and last > 0 else pagesize
        filter = self._get_run_filter_expr(**kwargs) if _filter_on_server else None

        client_kwargs = _generate_client_kwargs(has_query_params=True, orderby=order_by_expression, filter=filter,
                                                top=top, caller=caller, custom_headers=custom_headers,
                                                is_paginated=True)
        client_kwargs.update(kwargs)

        run_dtos = self._execute_with_experiment_arguments(self._client.run.get_by_query, **client_kwargs)

        return run_dtos if _filter_on_server else self._client_filter(run_dtos, **kwargs)

    def get_runs_by_run_ids(self, run_ids=None, page_size=DEFAULT_PAGE_SIZE, order_by=None,
                            caller=None, custom_headers=None):
        """
        Get detail of all runs of an experiment
        :param run_ids: list of run ids
        :type run_ids: [str]
        :param page_size: number of dto returned by one request (optional)
        :type page_size: int
        :param order_by: keys to sort return values, ('sort_key', 'asc'/'desc')(optional)
        :type order_by: tuple (str, str)
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return: a generator of ~_restclient.models.RunDto
        """
        order_by_expression = _validate_order_by(order_by) if order_by else [ORDER_BY_RUNID_EXPRESSION]
        filter = get_run_ids_filter_expression(run_ids) if run_ids is not None else None
        kwargs = _generate_client_kwargs(has_query_params=True, top=page_size, orderby=order_by_expression,
                                         filter=filter, caller=caller, custom_headers=custom_headers,
                                         is_paginated=True)

        return self._execute_with_experiment_arguments(self._client.run.get_by_query, **kwargs)

    def get_metrics(self, run_id=None, page_size=DEFAULT_PAGE_SIZE, order_by=None,
                    merge_strategy_type=None, caller=None, metric_types=None, after_timestamp=None,
                    custom_headers=None, name=None):
        """
        Get run_metrics of an experiment
        :param run_id: run id (optional)
        :type run_id: str
        :param page_size: number of dto returned by one request (optional)
        :type page_size: int
        :param order_by: keys to sort return values, ('sort_key', 'asc'/'desc')(optional)
        :type order_by: tuple (str, str)
        :param mergestrategytype: Possible values include: 'Default', 'None',
         'MergeToVector'
        :type mergestrategytype: str
        :param caller: caller function name (optional)
        :type caller: str
        :param metric_types: metric types to get (optional)
        :type metric_types: [str]
        :param after_timestamp: earliest timestamp of metrics to get (optional)
        :type after_timestamp: datetime
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param name: The name of the metric.
        :type name: str
        :return: a generator of ~_restclient.models.RunMetricDto
        """
        metrics_filter = get_filter_expression(run_ids=[run_id], metric_types=metric_types, metric_name=name,
                                               after_timestamp=after_timestamp)
        order_by_expression = _validate_order_by(order_by)
        kwargs = _generate_client_kwargs(top=page_size, orderby=order_by_expression, caller=caller,
                                         mergestrategytype=merge_strategy_type,
                                         custom_headers=custom_headers, is_paginated=True,
                                         has_query_params=True, filter=metrics_filter)

        return self._execute_with_experiment_arguments(self._client.run_metric.get_by_query, **kwargs)

    def get_metrics_by_run_ids(self, run_ids=None, page_size=DEFAULT_PAGE_SIZE, order_by=None,
                               merge_strategy_type=None, caller=None, metric_types=None, after_timestamp=None,
                               custom_headers=None, name=None):
        """
        Get run_metrics of multiple runs
        :param run_ids: run ids(optional)
        :type run_ids: [str]
        :param page_size: number of dto returned by one request (optional)
        :type page_size: int
        :param order_by: keys to sort return values, ('sort_key', 'asc'/'desc')(optional)
        :type order_by: tuple (str, str)
        :param mergestrategytype: Possible values include: 'Default', 'None', 'MergeToVector'
        :type mergestrategytype: str
        :param caller: caller function name (optional)
        :type caller: str
        :param metric_types: metric types to get (optional)
        :type metric_types: [str]
        :param after_timestamp: earliest timestamp of metrics to get (optional)
        :type after_timestamp: datetime
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param name: The name of the metric.
        :type name: str
        :return: a generator of ~_restclient.models.RunMetricDto
        """
        order_by_expression = _validate_order_by(order_by) if order_by else [ORDER_BY_RUNID_EXPRESSION]
        metrics_filter = get_filter_expression(run_ids=run_ids, metric_types=metric_types, metric_name=name,
                                               after_timestamp=after_timestamp)
        kwargs = _generate_client_kwargs(top=page_size, orderby=order_by_expression, caller=caller,
                                         mergestrategytype=merge_strategy_type,
                                         custom_headers=custom_headers, is_paginated=True,
                                         has_query_params=True, filter=metrics_filter)

        return self._execute_with_experiment_arguments(self._client.run_metric.get_by_query, **kwargs)

    def _execute_with_experiment_arguments(self, func, *args, **kwargs):
        return self._execute_with_arguments(func, copy.deepcopy(self._experiment_arguments), *args, **kwargs)

    def _execute_with_experimentid_arguments(self, func, *args, **kwargs):
        if self._experiment_id is None:
            raise ValueError("experiment_id should not be None.")
        return self._execute_with_arguments(func,
                                            copy.deepcopy(
                                                self._experiment_arguments_with_experiment_id),
                                            *args, **kwargs)

    def _combine_with_experiment_paginated_dto(self, func, count_to_download=0, *args, **kwargs):
        return self._combine_paginated_base(self._execute_with_experiment_arguments,
                                            func,
                                            count_to_download,
                                            *args,
                                            **kwargs)

    def _filter_run_on_created_after(run_dto, created_after):
        return run_dto.created_utc >= created_after

    def _filter_run_on_status(run_dto, status):
        return run_dto.status == status

    def _filter_run_on_type(run_dto, type):
        return run_dto.run_type == type

    def _filter_run_on_type_v2_orchestrator(run_dto, orchestrator):
        return run_dto.run_type_v2.orchestrator == orchestrator

    def _filter_run_on_type_v2_traits(run_dto, trait):
        return trait in run_dto.run_type_v2.traits

    def _filter_run_on_tags(run_dto, tags):
        if isinstance(tags, six.text_type) and tags in run_dto.tags.keys():
            return True
        elif isinstance(tags, dict):
            if set(tags.items()).issubset(run_dto.tags.items()):
                return True
        return False

    def _filter_run_on_props(run_dto, props):
        if isinstance(props, six.text_type) and props in run_dto.properties.keys():
            return True
        elif isinstance(props, dict):
            if set(props.items()).issubset(run_dto.properties.items()):
                return True
        return False

    def _filter_run_on_include_children(run_dto, include_children):
        is_parent = run_dto.parent_run_id
        return is_parent is None or include_children

    def _filter_run_on_target_name(run_dto, target_name):
        return run_dto.target == target_name

    _run_client_filter_mapping = {
        "tags": _filter_run_on_tags,
        "properties": _filter_run_on_props,
        "runtype": _filter_run_on_type,
        'orchestrator': _filter_run_on_type_v2_orchestrator,
        'trait': _filter_run_on_type_v2_traits,
        "created_after": _filter_run_on_created_after,
        "status": _filter_run_on_status,
        "include_children": _filter_run_on_include_children,
        "target_name": _filter_run_on_target_name
    }

    def _client_filter(self, run_dtos, **kwargs):
        filter_funcs = {}
        for filter_type, filter_val in kwargs.items():
            if filter_val is None:
                self._logger.debug("Skipping filter %s for None val", filter_type)
                continue

            filter_func = ExperimentClient._run_client_filter_mapping.get(filter_type, None)
            if filter_func is None:
                self._logger.warning(
                    "Received unknown filter type: {0} on {1}".format(filter_type, filter_val))
            else:
                self._logger.debug("Getting filter %s for %s", filter_func, filter_val)
                filter_funcs[filter_func] = filter_val

        for run_dto in run_dtos:
            skip = False
            for func, val in filter_funcs.items():
                self._logger.debug("client filtering %s on %s", run_dto, val)
                if not func(run_dto, val):
                    skip = True
            if not skip:
                yield run_dto

    def get_metrics_in_batches_by_run_ids(self, run_ids, metric_types=None, after_timestamp=None, name=None,
                                          merge_strategy_type=None, custom_headers=None):
        """Get multiple metrics by run history run ids.

        :param run_ids: Run Ids for the metrics to fetch. *Note* Best Metric value is retrieved
        and updated for these runs
        :type run_ids: [str]
        :param metric_types: the types of metrics to fetch for the runs (optional)
        :type metric_types: [str]
        :param after_timestamp: earliest timestamp of metrics to get (optional)
        :type after_timestamp: datetime
        :param name: The name of the metric.
        :type name: str
        :param merge_strategy_type: strategy for how metrics get merged
        :type merge_strategy_type: str
        :param custom_headers: custom headers to be added
        :type custom_headers: dict
        :return: Transformed metric data in a friendly format.
        :rtype: dict
        """

        # Short-circuit if there's no work to do
        if not run_ids:
            return {}

        def _batches(items, size):
            """Convert a list into batches of the specified size.

            :param items: The list of items to split into batches.
            :type items: list
            :param size: The number of items in each batch.
            :type size: int
            """
            for i in range(0, len(items), size):
                yield items[i:i + size]

        # Number of run ids per batch
        batch_size = 50

        # With hyperdrive/automl scenarios count of runs can get quite large and GET request limit may be reached
        # easily. We will need to group runs into batches and fetch the metrics based on defined degree
        # of parallelism.
        _batches = list(_batches(sorted(run_ids), batch_size))
        tasks = []

        common_headers = {'x-ms-client-session-id': _ClientSessionId}

        if custom_headers is not None:
            common_headers.update(custom_headers)

        for batch in _batches:
            result_as_generator = self.get_metrics_by_run_ids(
                run_ids=batch, custom_headers=common_headers,
                order_by=('RunId', 'asc'), metric_types=metric_types,
                after_timestamp=after_timestamp, name=name,
                merge_strategy_type="None")
            tasks.append(result_as_generator)

        return tasks
