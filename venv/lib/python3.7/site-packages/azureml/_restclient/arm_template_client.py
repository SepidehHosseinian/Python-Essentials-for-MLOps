# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------
"""Access Arm Template client"""

from .clientbase import ClientBase

from .utils import _generate_client_kwargs


class ArmTemplateClient(ClientBase):
    """
    Arm Template APIs

    :param host: The base path for the server to call.
    :type host: str
    :param auth: Client authentication
    :type auth: azureml.core.authentication.AbstractAuthentication
    :param subscription_id:
    :type subscription_id: str
    :param resource_group_name:
    :type resource_group_name: str
    """
    def __init__(self, service_context, host=None, **kwargs):
        """
        Constructor of the class.
        """
        self._service_context = service_context
        self._override_host = host
        super(ArmTemplateClient, self).__init__(service_context, **kwargs)

    @property
    def auth(self):
        return self._service_context.get_auth()

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_arm_template_restclient(
            host=self._override_host, user_agent=user_agent)

    def set_up_env(self, subscription_id, resource_group_name, arm_template_request_dto,
                   caller=None, custom_headers=None, is_async=False):
        """
        Create workspace using template API

        :param subscription_id: subscription id to create the workspace
        :type subscription_id: str
        :param resource_group_name: resourcegroup to create the workspace
        :type resource_group_name: str
        :param arm_template_request_dto: arm template request dto
        :type arm_template_request_dto: ArmTemplateDto
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
        rtype: None (is_async is False) or
            azureml._async.AsyncTask (is_async is True)
        """
        kwargs = _generate_client_kwargs(
            custom_headers=custom_headers, is_async=is_async, caller=caller)
        return self._execute_with_arguments(
            self._client.arm_template.env_set_up, [],
            subscription_id, resource_group_name, arm_template_request_dto, **kwargs)
