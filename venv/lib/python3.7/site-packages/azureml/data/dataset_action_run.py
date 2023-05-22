# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality that manages the execution of Dataset actions.

This module provides convenience methods for creating Dataset actions and get their results after completion.
"""

from azureml.data._dataset_deprecation import deprecated


class DatasetActionRun(object):
    """Manage the execution of Dataset actions.

    DatasetActionRun provides methods for monitoring the status of long running actions on datasets. It
    also provides a method to get the result of an action after completion.
    """

    # the auth token received from _auth.get_authentication_header is prefixed
    # with 'Bearer '. This is used to remove that prefix.
    _bearer_prefix_len = 7

    @deprecated('DatasetActionRun class')
    def __init__(self, workspace=None, dataset_id=None, action_id=None, action_request_dto=None):
        """Initialize a DatasetActionRun.

        :param workspace: The workspace the Dataset is in.
        :type workspace: azureml.core.Workspace
        :param dataset_id: The Dataset ID.
        :type dataset_id: str
        :param action_id: The Dataset action ID
        :type action_id: str
        :param action_request_dto: The action request dto.
        :type action_request_dto: azureml._restclient.models.action_result_dto
        """
        self._workspace = workspace
        self._dataset_id = dataset_id
        self._action_id = action_id
        self._action_request_dto = action_request_dto
        self._result = None

    def wait_for_completion(self, show_output=True, status_update_frequency=10):
        """Wait for the completion of Dataset action run.

        .. remarks::

            This is a synchronous method. Call this if you have triggered a long running action on a dataset
            and you want to wait for the action to complete before proceeding. This method writes the status of the
            action run in the logs periodically, with the interval between updates determined by the
            ``status_update_frequency`` parameter.

            The action returns when the action has completed. To inspect the result of the action, use
            :func:`get_result`.

        :param show_output: Indicates whether to print the output.
        :type show_output: bool
        :param status_update_frequency: The action run status update frequency in seconds.
        :type status_update_frequency: int
        """
        if self._result is not None:
            return
        DatasetActionRun._client()._wait_for_completion(
            workspace=self._workspace,
            dataset_id=self._dataset_id,
            action_id=self._action_id,
            show_output=show_output,
            status_update_frequency=status_update_frequency)

    def get_result(self):
        """Get the result of completed Dataset action run.

        :return: The Dataset action result.
        :rtype: typing.Union[azureml.dataprep.DataProfile, None]
        """
        if self._result is not None:
            return self._result

        if self._action_request_dto.action_type == 'profile':
            return DatasetActionRun._client()._get_profile(
                workspace=self._workspace,
                dataset_id=self._dataset_id,
                action_id=self._action_id,
                datastore_name=self._action_request_dto.datastore_name)
        elif self._action_request_dto.action_type == 'diff':
            return DatasetActionRun._client()._get_profile_diff_result(
                workspace=self._workspace,
                action_id=self._action_id,
                dataset_id=self._dataset_id,
                action_request_dto=self._action_request_dto)
        elif self._action_request_dto.action_type == 'datasetdiff':
            return DatasetActionRun._client()._get_diff_result(
                workspace=self._workspace,
                action_id=self._action_id,
                dataset_id=self._dataset_id,
                action_request_dto=self._action_request_dto)
        else:
            return None

    @staticmethod
    def _client():
        """Get a Dataset client.

        :return: Returns the client.
        :rtype: DatasetClient
        """
        from ._dataset_client import _DatasetClient
        return _DatasetClient
