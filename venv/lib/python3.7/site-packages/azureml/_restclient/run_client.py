# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Access a experiment client"""
import copy
import logging

from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import OnlySupportedServiceSideFiltering
from .contracts.events import (create_start_event, create_completed_event,
                               create_failed_event, create_canceled_event, create_cancel_requested_event,
                               create_heartbeat_event,
                               create_error_event,
                               create_warning_event)
from .constants import DEFAULT_PAGE_SIZE
from ._odata.constants import ORDER_BY_STARTTIME_EXPRESSION
from .experiment_client import ExperimentClient
from .models.create_run_dto import CreateRunDto
from .models.query_params_dto import QueryParamsDto
from .models.batch_add_or_modify_run_request_dto import BatchAddOrModifyRunRequestDto
from .models.batch_event_command_dto import BatchEventCommandDto
from .utils import _generate_client_kwargs, _validate_order_by
from ._hierarchy.runs import Tree
from ._odata.expressions import and_join


module_logger = logging.getLogger(__name__)


class RunClient(ExperimentClient):
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
    :param run_id:
    :type run_id: str
    """

    def __init__(self,
                 service_context,
                 experiment_name,
                 run_id,
                 **kwargs):
        """
        Constructor of the class.
        """
        super(RunClient, self).__init__(service_context,
                                        experiment_name,
                                        **kwargs)
        self._run_id = run_id
        self._run_arguments = copy.deepcopy(self._experiment_arguments)
        self._run_arguments.append(self._run_id)
        self._run_arguments_with_experiment_id = copy.deepcopy(self._experiment_arguments_with_experiment_id)
        self._run_arguments_with_experiment_id.append(self._run_id)

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_run_history_restclient(user_agent=user_agent)

    def get_run(self, **kwargs):
        """
        Get detail of a run by its run_id
        This function could also be called from the super class,
        ExperimentClient, for a specific run_id
        """
        return super(RunClient, self).get_run(self._run_id, **kwargs)

    def get_token(self):
        """
        Get an azure machine learning token for the run scope
        :return: token and token expiry information.
        :rtype: azureml._restclient.models.token_result.TokenResult
        """
        return self._execute_with_run_arguments(self._client.run.get_token)

    def get_runstatus(self, caller=None, custom_headers=None, is_async=False):
        """
        Get status details of a run by its run_id
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
        rtype: ~_restclient.models.RunDetailsDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            is_async=is_async, caller=caller, custom_headers=custom_headers)

        return self._execute_with_run_arguments(self._client.run.get_details, **kwargs)

    def create_child_run(self, child_run_id, script_name=None,
                         target=None, run_name=None, caller=None, custom_headers=None, is_async=False,
                         properties=None, tags=None):
        """
        Create a child run
        :param child_run_id: child_run_id(required)
        :type child_run_id: str
        :param script_name: script name
        :type script_name: str
        :param target: run target
        :type target: str
        :param run_name: run_name
        :type run_name: str
        :param is_async: execute request asynchronously
        :type is_async: bool (optional)
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param properties: Optional initial properties of a run
        :type properties: dict[str]
        :param tags: Optional initial tags on a run
        :type tags: dict[str]
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.RunDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            is_async=is_async, caller=caller, custom_headers=custom_headers)

        create_run_dto = CreateRunDto(run_id=child_run_id,
                                      parent_run_id=self._run_id,
                                      script_name=script_name,
                                      target=target,
                                      name=run_name,
                                      status='NotStarted',
                                      properties=properties,
                                      tags=tags)

        if self._experiment_id:
            return self._execute_with_experimentid_arguments(self._client.run.patch_by_exp_id,
                                                             run_id=child_run_id,
                                                             create_run_dto=create_run_dto,
                                                             **kwargs)
        return self._execute_with_experiment_arguments(self._client.run.patch,
                                                       run_id=child_run_id,
                                                       create_run_dto=create_run_dto,
                                                       **kwargs)

    def batch_create_child_runs(self, child_runs, caller=None, custom_headers=None, is_async=False):
        """
        Create one or many child runs
        :param child_runs: list of runs to create
        :type child_runs: [CreateRunDto]
        :param is_async: execute request asynchronously
        :type is_async: bool (optional)
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return:
            the return type is based on is_async parameter
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.BatchAddOrModifyRunResultDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            is_async=is_async, caller=caller, custom_headers=custom_headers)

        request_dto = BatchAddOrModifyRunRequestDto(child_runs)
        return self._execute_with_experiment_arguments(self._client.run.batch_add_or_modify,
                                                       request_dto,
                                                       **kwargs)

    def get_child_runs(self, root_run_id, recursive=False, _filter_on_server=False,
                       page_size=DEFAULT_PAGE_SIZE, order_by=None,
                       caller=None, custom_headers=None, **kwargs):
        """
        Get child runs by current run_id
        :param root_run_id: optimization id for hierarchy(required)
        :type root_run_id: str
        :param recursive: fetch grandchildren and further descendants(required)
        :type recursive: bool
        :param page_size: number of dto returned by one request (optional)
        :type page_size: int
        :param order_by: keys to sort return values, ('sort_key', 'asc'/'desc')(optional)
        :type order_by: tuple (str, str)
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return: list of dictionary whose keys are property of ~_restclient.models.RunDto
        """
        order_by_expression = _validate_order_by(order_by) if order_by else [ORDER_BY_STARTTIME_EXPRESSION]
        client_kwargs = _generate_client_kwargs(top=page_size, orderby=order_by_expression, caller=caller,
                                                custom_headers=custom_headers, is_paginated=True)
        client_kwargs.update(kwargs)
        # TODO: _restclient shouldn't depend on core
        if recursive and _filter_on_server and root_run_id != self._run_id:
            azureml_error = AzureMLError.create(
                OnlySupportedServiceSideFiltering
            )
            raise AzureMLException._with_error(azureml_error)
        elif recursive:
            _filter_on_server = root_run_id == self._run_id or _filter_on_server
            filter_expression = self._get_run_filter_expr(**kwargs) if _filter_on_server else None
            root_filter = 'RootRunId eq {0}'.format(root_run_id)
            exclude_parent_filter = 'RunId ne {0}'.format(self._run_id)
            full_filter = and_join([root_filter, filter_expression]) if filter_expression else root_filter
            full_filter = and_join([full_filter, exclude_parent_filter]) if _filter_on_server else full_filter

            query_params = QueryParamsDto(filter=full_filter)
            run_dtos = self._execute_with_experiment_arguments(self._client.run.get_by_query,
                                                               query_params=query_params,
                                                               **client_kwargs)

            if _filter_on_server:
                return run_dtos

            # Filter out nodes outside of the desired sub tree
            run_hierarchy = Tree(run_dtos)
            sub_tree_run_dtos = run_hierarchy.get_subtree_dtos(self._run_id)
            return self._client_filter(sub_tree_run_dtos, **kwargs)

        else:
            run_dtos = self._execute_with_run_arguments(self._client.run.get_child, **client_kwargs)
            return run_dtos if _filter_on_server else self._client_filter(run_dtos, **kwargs)

    def patch_run(self, create_run_dto, caller=None, custom_headers=None, is_async=False):
        """
        Patch a run to the run history client
        :param create_run_dto: a new run object(required)
        :type create_run_dto: ~_restclient.models.CreateRunDto
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param is_async: execute request asynchronously
        :type is_async: bool (optional)
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
            return self._execute_with_run_expid_arguments(self._client.run.patch_by_exp_id,
                                                          create_run_dto=create_run_dto,
                                                          **kwargs)
        return self._execute_with_run_arguments(self._client.run.patch,
                                                create_run_dto=create_run_dto,
                                                **kwargs)

    def delete_run_tags(self, tags, caller=None, custom_headers=None, is_async=False):
        """
        Delete tags on a run to the run history client
        :param tags: list of tag keys to be deleted(required)
        :type tags: [str]
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param is_async: execute request asynchronously
        :type is_async: bool (optional)
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.RunDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            is_async=is_async, caller=caller, custom_headers=custom_headers)

        return self._execute_with_run_arguments(self._client.run.delete_tags, tags=tags, **kwargs)

    def post_event(self, event, caller=None, custom_headers=None, is_async=False):
        """
        Post an event of a run by its run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param event: event to update (required)
        :type event: models.BaseEvent
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(event_message=event, caller=caller,
                                         custom_headers=custom_headers, is_async=is_async)

        return self._execute_with_run_arguments(self._client.events.post, **kwargs)

    def batch_post_event_start(self, run_ids, caller=None, custom_headers=None, is_async=False):
        """
        Post one or many run-started-events
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param run_ids: runs to start (required)
        :type run_ids: [str]
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: BatchEventCommandResultDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(caller=caller, custom_headers=custom_headers, is_async=is_async)

        events = [create_start_event(run_id) for run_id in run_ids]
        request_dto = BatchEventCommandDto(events)
        return self._execute_with_experiment_arguments(self._client.events.batch_post, request_dto, **kwargs)

    def post_event_start(self, caller=None, custom_headers=None, is_async=False):
        """
        Post run-started-event for this run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_header: dict
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_start_event(self._run_id)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_completed(self, caller=None, custom_headers=None, is_async=False):
        """
        Post run-completed-event for this run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_completed_event(self._run_id)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_failed(self, caller=None, custom_headers=None, is_async=False):
        """
        Post run-failed-event for this run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_failed_event(self._run_id)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_canceled(self, caller=None, custom_headers=None, is_async=False):
        """
        Post run-canceled-event by its run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_canceled_event(self._run_id)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_cancel_requested(self, caller=None, custom_headers=None, is_async=False):
        """
        Post run-cancel-requested-event by its run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_cancel_requested_event(self._run_id)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_heartbeat(self, time, caller=None, custom_headers=None, is_async=False):
        """
        Post run-heartbeat-event by its run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param run_id: run id (required)
        :type run_id: str
        :param time: timeout in seconds (required)
        :type time: int
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_heartbeat_event(self._run_id, time)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_error(self, error_details, caller=None, custom_headers=None, is_async=False):
        """
        Post run-error-event by its run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param error_details: error_details object
        :type error_details: dict
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_error_event(self._run_id, error_details)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def post_event_warning(self, source, message, caller=None, custom_headers=None, is_async=False):
        """
        Post run-warning-event by its run_id
        This method makes a synchronous call by default. To make an
        asynchronous call, please pass is_async=True
        :param source: warning source
        :type source: str
        :param message: warning message
        :type message: str
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        event = create_warning_event(self._run_id, source, message)
        return self.post_event(event, is_async=is_async, caller=caller, custom_headers=custom_headers)

    def wait_for_metrics_ingest(self, timeout_seconds=600, caller=None, custom_headers=None):
        """
        Wait for background metrics ingest to complete
        :param timeout_seconds: seconds after which to stop waiting
        :type timeout_seconds: int
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return: when the metrics ingest is complete
        """
        call_kwargs = {
            'raw': True
        }

        # initial response could be 200 or 202
        initial_response = self._execute_with_run_arguments(self._client.run_metric.wait_on_ingest, **call_kwargs)

        from .polling import AzureMLPolling
        from msrest.polling.poller import LROPoller

        # "AzureML polling" is a name for the 202/200/location-header contract
        arm_poller = AzureMLPolling(
            timeout=5,  # timeout here is actually the delay between polls, bad name
            lro_options={'final-state-via': 'location'}
        )

        # we never care about reading the body
        def deserialization_callback(response):
            return None

        poller = LROPoller(
            self._client.run_metric._client,
            initial_response,
            deserialization_callback,
            arm_poller
        )

        # this call blocks until the async operation returns 200
        poller.result(timeout_seconds)

    def _execute_with_run_arguments(self, func, *args, **kwargs):
        return self._execute_with_arguments(func, copy.deepcopy(self._run_arguments), *args, **kwargs)

    def _execute_with_run_expid_arguments(self, func, *args, **kwargs):
        return self._execute_with_arguments(
            func,
            copy.deepcopy(self._run_arguments_with_experiment_id), *args, **kwargs)

    def _execute_with_workspace_run_arguments(self, func, *args, **kwargs):
        workspace_run_arguments = copy.deepcopy(self._workspace_arguments)
        run_id = kwargs.pop("run_id", self._run_id)
        workspace_run_arguments.append(run_id)
        return self._execute_with_arguments(func, workspace_run_arguments, *args, **kwargs)

    def _combine_with_run_paginated_dto(self, func, count_to_download=0, *args, **kwargs):
        return self._combine_paginated_base(self._execute_with_run_arguments,
                                            func,
                                            count_to_download,
                                            *args,
                                            **kwargs)
