# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------
"""Access workspace client"""
import copy

from azureml._restclient.models import ModifyExperimentDto, DeleteTagsCommandDto
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import FailedIdWithinSeconds
from azureml.exceptions import UserErrorException
from .constants import DEFAULT_PAGE_SIZE
from .clientbase import ClientBase

from ._odata.constants import ORDER_BY_CREATEDTIME_EXPRESSION
from ._odata.experiments import get_filter_expression

from .utils import _generate_client_kwargs, _validate_order_by
from .exceptions import ServiceException
from ._odata.expressions import and_join
from ._odata.runs import (get_filter_run_tags, get_filter_run_properties, get_filter_run_type,
                          get_filter_run_type_v2_orchestrator, get_filter_run_type_v2_traits,
                          get_filter_run_created_after, get_filter_run_status, get_filter_include_children,
                          get_filter_run_target_name)


class WorkspaceClient(ClientBase):
    """
    Run History APIs

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
    """

    def __init__(self, service_context, host=None, **kwargs):
        """
        Constructor of the class.
        """
        self._service_context = service_context
        self._override_host = host
        self._workspace_arguments = [self._service_context.subscription_id,
                                     self._service_context.resource_group_name,
                                     self._service_context.workspace_name]
        super(WorkspaceClient, self).__init__(**kwargs)

    @property
    def auth(self):
        return self._service_context.get_auth()

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_run_history_restclient(
            host=self._override_host, user_agent=user_agent)

    def get_cluster_url(self):
        """get service url"""
        return self._host

    def get_workspace_uri_path(self):
        return self._service_context._get_workspace_scope()

    def _execute_with_workspace_arguments(self, func, *args, **kwargs):
        return self._execute_with_arguments(func, copy.deepcopy(self._workspace_arguments), *args, **kwargs)

    def get_or_create_experiment(self, experiment_name, is_async=False):
        """
        get or create an experiment by name
        :param experiment_name: experiment name (required)
        :type experiment_name: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async_task.AsyncTask object
            If parameter is_async is False or missing,
            return: ~_restclient.models.ExperimentDto
        """

        # Client Get, Create on NotFound
        try:
            return self.get_experiment(experiment_name, is_async)
        except ServiceException as e:
            if e.status_code == 404:
                return self._execute_with_workspace_arguments(self._client.experiment.create,
                                                              experiment_name=experiment_name,
                                                              is_async=is_async)
            else:
                raise

    def get_run(self, run_id, is_async=False, caller=None, custom_headers=None):
        """
        Get the specified run within the workspace
        :param run_id: runId (required)
        :type run_id: str
        :param is_async: execute request asynchronously
        :type is async: bool
        :param caller: caller function name (optional)
        :type caller: optional[string]
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: optional[dict]
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async_task.AsyncTask object
            If parameter is_async is False or missing,
            return: ~_restclient.models.RunDto
        """

        kwargs = _generate_client_kwargs(is_async=is_async, caller=caller, custom_headers=custom_headers)
        return self._execute_with_workspace_arguments(self._client.run.get_workspace_run,
                                                      run_id=run_id,
                                                      **kwargs)

    def list_experiments(self, last=None, order_by=None, experiment_name=None, view_type=None, tags=None):
        """
        list all experiments
        :return: a generator of ~_restclient.models.ExperimentDto
        """

        kwargs = {}
        if last is not None:
            order_by_expression = _validate_order_by(order_by) if order_by else [ORDER_BY_CREATEDTIME_EXPRESSION]
            kwargs = _generate_client_kwargs(top=last, orderby=order_by_expression)
            # TODO: Doesn't work
            raise NotImplementedError("Cannot limit experiment list")

        filter_expression = get_filter_expression(experiment_name=experiment_name, tags=tags)
        filter_expression = None if filter_expression == "" else filter_expression

        kwargs = _generate_client_kwargs(has_query_params=True, filter=filter_expression,
                                         generate_experiment_query_params=True, view_type=view_type)

        return self._execute_with_workspace_arguments(self._client.experiment.get_by_query,
                                                      is_paginated=True,
                                                      **kwargs)

    def delete_experiment(self, experiment_id, timeout_seconds=600):
        """
        delete empty experiment by experiment_id
        :return: when the delete operation is complete
        """
        call_kwargs = {
            'raw': True
        }

        # initial response could be 200 or 202
        initial_response = self._execute_with_workspace_arguments(self._client.experiment.delete,
                                                                  experiment_id=experiment_id,
                                                                  **call_kwargs)

        from .polling import AzureMLPolling
        from msrest.polling.poller import LROPoller

        # "AzureML polling" is a name for the 202/200/location-header contract
        arm_poller = AzureMLPolling(
            timeout=5,  # timeout here is actually the delay between polls, bad name
            lro_options={'final-state-via': 'location'}
        )

        # raise an exception to the user when the timeout expires and still got a 202
        def deserialization_callback(response):
            return 1 if response is not None and response.status_code == 202 else 0

        poller = LROPoller(
            self._client.experiment._client,
            initial_response,
            deserialization_callback,
            arm_poller
        )

        # this call blocks until the async operation returns 200
        result = poller.result(timeout_seconds)
        if result == 1:
            azureml_error = AzureMLError.create(
                FailedIdWithinSeconds, experiment_id=experiment_id, timeout_seconds=timeout_seconds
            )
            raise AzureMLException._with_error(azureml_error)

    def get_runs_by_compute(self,
                            compute_name,
                            last=0,
                            page_size=DEFAULT_PAGE_SIZE,
                            caller=None,
                            custom_headers=None,
                            **kwargs):
        """
        Get detail of all runs of an experiment
        :param compute_name: the name of the compute to get runs from
        :type compute_name: str
        :param last: the number of latest runs to return (optional, default to all)
        :type last: int
        :param page_size: number of dto returned by one request (optional, default 500)
        :type page_size: int
        :param caller: caller function name (optional)
        :type caller: str
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: dict
        :return: a generator of ~_restclient.models.RunDto
        """

        # different from experiment.get_runs in that
        # filter on server is always true and there is no order experession
        client_kwargs = _generate_client_kwargs(caller=caller,
                                                custom_headers=custom_headers,
                                                is_paginated=True)
        client_kwargs.update(kwargs)

        pagesize = DEFAULT_PAGE_SIZE if page_size < 1 else page_size
        top = last if last < pagesize and last > 0 else pagesize
        filter_expression = self._get_run_filter_expr(**kwargs)

        run_dtos = self._execute_with_compute_arguments(self._client.run.list_by_compute,
                                                        compute_name,
                                                        total_count=last,
                                                        top=top,
                                                        filter=filter_expression,
                                                        **client_kwargs)

        return run_dtos

    def _execute_with_compute_arguments(self, func, compute_name, *args, **kwargs):
        compute_arguments = copy.deepcopy(self._workspace_arguments)
        compute_arguments.append(compute_name)
        return self._execute_with_arguments(func, compute_arguments, *args, **kwargs)

    _run_filter_mapping = {
        'tags': get_filter_run_tags,
        'properties': get_filter_run_properties,
        'runtype': get_filter_run_type,
        'orchestrator': get_filter_run_type_v2_orchestrator,
        'trait': get_filter_run_type_v2_traits,
        'created_after': get_filter_run_created_after,
        "status": get_filter_run_status,
        "include_children": get_filter_include_children,
        "target_name": get_filter_run_target_name
    }

    def _get_run_filter_expr(self, **kwargs):
        exprs = []
        for filter_type, filter_val in kwargs.items():
            if filter_val is None:
                self._logger.debug("Skipping filter %s for None val", filter_type)
                continue
            filter_func = WorkspaceClient._run_filter_mapping.get(filter_type, None)
            if filter_func is None:
                self._logger.warning(
                    "Received unknown filter type: {0} on {1}".format(filter_type, filter_val))
            else:
                self._logger.debug("Getting filter %s for %s", filter_func, filter_val)
                filter_query = filter_func(filter_val)
                if filter_query is not None:
                    exprs.append(filter_query)
        return None if len(exprs) < 1 else and_join(exprs)

    def get_experiment(self, experiment_name, is_async=False):
        """
        get experiment by name
        :param experiment_name: experiment name (required)
        :type experiment_name: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async_task.AsyncTask object
            If parameter is_async is False or missing,
            return: ~_restclient.models.ExperimentDto
        """

        return self._execute_with_workspace_arguments(self._client.experiment.get,
                                                      experiment_name=experiment_name,
                                                      is_async=is_async)

    def get_experiment_by_id(self, experiment_id, is_async=False):
        """
        get experiment by id
        :param experiment_id: experiment id (required)
        :type experiment_id: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :return:
            If is_async parameter is True,
            the request is called asynchronously.
            The method returns azureml._async_task.AsyncTask object
            If parameter is_async is False or missing,
            return: ~_restclient.models.ExperimentDto
        """

        return self._execute_with_workspace_arguments(self._client.experiment.get_by_id,
                                                      experiment_id=experiment_id,
                                                      is_async=is_async)

    def archive_experiment(self, experiment_id, caller=None, custom_headers=None, is_async=False):
        """
        Archive the experiment
        :param experiment_id: experiment id (required)
        :type experiment_id: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: optional[string]
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: optional[dict]
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.ExperimentDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        modify_experiment_dto = ModifyExperimentDto(archive=True)
        return self.update_experiment(experiment_id, modify_experiment_dto, caller, custom_headers, is_async)

    def reactivate_experiment(self, experiment_id, new_name=None, caller=None, custom_headers=None, is_async=False):
        """
        Reactivate an archived experiment
        :param experiment_id: experiment id (required)
        :type experiment_id: str
        :param new_name: new experiment name (optional)
        :type new_name: str
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: optional[string]
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: optional[dict]
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.ExperimentDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        if new_name is not None:
            raise UserErrorException("Cannot rename an experiment. If the archived experiment name conflicts"
                                     " with an active experiment name, you can delete the active experiment"
                                     " before unarchiving this experiment.")
        modify_experiment_dto = ModifyExperimentDto(archive=False)
        return self.update_experiment(experiment_id, modify_experiment_dto, caller, custom_headers, is_async)

    def set_tags(self, experiment_id, tags=None, caller=None, custom_headers=None, is_async=False):
        """
        Modify the tags on an experiment
        :param experiment_id: experiment id (required)
        :type experiment_id: str
        :param tags: tags to modify (optional)
        :type tags: dict[str]
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: optional[string]
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: optional[dict]
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.ExperimentDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        sanitized_tags = self._sanitize_tags(tags)
        modify_experiment_dto = ModifyExperimentDto(tags=sanitized_tags)
        return self.update_experiment(experiment_id, modify_experiment_dto, caller, custom_headers, is_async)

    def update_experiment(self, experiment_id, modify_experiment_dto,
                          caller=None, custom_headers=None, is_async=False):
        """
        Update the experiment
        :param experiment_id: experiment id (required)
        :type experiment_id: str
        :param modify_experiment_dto: modify experiment dto
        :type modify_experiment_dto: ModifyExperimentDto
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: optional[string]
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: optional[dict]
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.ExperimentDto (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            modify_experiment_dto=modify_experiment_dto,
            is_async=is_async, caller=caller, custom_headers=custom_headers)
        return self._execute_with_workspace_arguments(
            self._client.experiment.update, experiment_id=experiment_id, **kwargs)

    def delete_experiment_tags(self, experiment_id, tags, caller=None, custom_headers=None, is_async=False):
        """
        Delete the specified tags from the experiment
        :param experiment_id: experiment id (required)
        :type experiment_id: str
        :param tags: tag keys to delete
        :type tags: [str]
        :param is_async: execute request asynchronously
        :type is_async: bool
        :param caller: caller function name (optional)
        :type caller: optional[string]
        :param custom_headers: headers that will be added to the request (optional)
        :type custom_headers: optional[dict]
        :return:
            the return type is based on is_async parameter.
            If is_async parameter is True,
            the request is called asynchronously.
        rtype: ~_restclient.models.DeleteExperimentTagsResult (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        if tags is None:
            return
        tags = DeleteTagsCommandDto(tags)
        kwargs = _generate_client_kwargs(is_async, caller=caller, custom_headers=custom_headers)

        return self._execute_with_workspace_arguments(
            self._client.experiment.delete_tags, experiment_id=experiment_id, delete_tags_command_dto=tags, **kwargs)

    def _sanitize_tags(self, tag_or_prop_dict):
        # type: (...) -> {str}
        ret_tags = {}
        # dict comprehension would be nice but logging suffers without more functions
        for key, val in tag_or_prop_dict.items():
            if not isinstance(val, (str, type(None))):  # should be six.str/basestring or something
                self._logger.warn('Converting non-string tag to string: (%s: %s)', key, val)
                ret_tags[key] = str(val)
            else:
                ret_tags[key] = val
        return ret_tags
