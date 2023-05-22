# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Class for collecting summary statistics on the data produced by a Dataflow.

Functionality in this module includes collecting information regarding which run produced the
profile, whether the profile is stale or not.
"""
from azureml._common.exceptions import AzureMLException
from azureml._restclient.models import ActionRequestDto
from azureml.data._dataset_rest_helper import _restclient, _custom_headers
from azureml.data._loggerfactory import _LoggerFactory, track
from azureml.data.constants import _ACTION_TYPE_PROFILE, _LEGACY_DATASET_ID
from azureml._base_sdk_common._docstring_wrapper import experimental


_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = _LoggerFactory.get_logger(__name__)
    return _logger


@experimental
class DatasetProfile:
    """A DatasetProfile collects summary statistics on the data produced by a Dataflow.

    :param saved_dataset_id: The id of the dataset on which profile is computed.
    :type saved_dataset_id: str
    :param run_id: The run id for the experiment which is used to compute the profile.
    :type run_id: str
    :param experiment_name: The name of the submitted experiment used to compute the profile.
    :type experiment_name: str
    :param workspace: Workspace which the profile run belongs to.
        See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace
        for more information on workspaces.
    :type workspace: azureml.core.workspace.Workspace
    :param profile: Profile result from the latest profile run of type DataProfile.
    :type profile: azureml.dataprep.DataProfile
    """

    @track(_get_logger, custom_dimensions={'app_name': 'DatasetProfile'})
    def __init__(self, saved_dataset_id, run_id, experiment_name, workspace, profile):
        """Create DatasetProfile object.

        :param saved_dataset_id: The id of the dataset on which profile is computed.
        :type saved_dataset_id: str
        :param run_id: The run id for the experiment which is used to compute the profile.
        :type run_id: str
        :param experiment_name: The name of the submitted experiment used to compute the profile.
        :type experiment_name: str
        :param workspace: Workspace which the profile run belongs to.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace
            for more information on workspaces.
        :type workspace: azureml.core.workspace.Workspace
        :param profile: Profile result from the latest profile run of type DataProfile.
        :type profile: azureml.dataprep.DataProfile
        """
        if saved_dataset_id is None:
            _get_logger().error('saved_dataset_id is none')
            raise AzureMLException("Unable to fetch profile results. Please submit a new profile run.")
        if run_id is None:
            _get_logger().error('run_id is none')
            raise AzureMLException("Unable to fetch profile results. Please submit a new profile run.")
        if experiment_name is None:
            _get_logger().error('experiment_name is none')
            raise AzureMLException("Unable to fetch profile results. Please submit a new profile run.")
        if workspace is None:
            _get_logger().error('workspace is none')
            raise AzureMLException("Unable to fetch profile results. Please submit a new profile run.")
        if profile is None:
            _get_logger().error('profile is none')
            raise AzureMLException("Unable to fetch profile results. Please submit a new profile run.")

        self._saved_dataset_id = saved_dataset_id
        self._run_id = run_id
        self._experiment_name = experiment_name
        self._workspace = workspace
        self._profile = profile

    @experimental
    def get_producing_run(self):
        """Return the experiment Run object of type `Run` that produced this profile.

        :return: The submitted experiment run for this profile run.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)
            for more information on run.
        :rtype: azureml.core.Run
        """
        from azureml.core import Experiment, get_run
        experiment = Experiment(self._workspace, self._experiment_name)
        return get_run(experiment, self._run_id)

    @experimental
    def is_stale(self):
        """Return boolean to describe whether the computed profile is stale or not.

        A Profile is considered to be stale if there is changed in underlying data after the
        profile is computed.
        - if the data source change cannot be detected, TypeError is raised.
        - if the data source was changed after submitting the profile run, the flag will be True;
        - otherwise, the profile matches current data, and the flag will be False.

        :return: boolean to describe whether the computed profile is stale or not.
        :rtype: bool
        """
        from azureml.core import Dataset
        dataset = Dataset.get_by_id(self._workspace, id=self._saved_dataset_id)
        workspace = dataset._ensure_workspace(self._workspace)

        request_dto = ActionRequestDto(
            action_type=_ACTION_TYPE_PROFILE,
            saved_dataset_id=dataset._ensure_saved(workspace),
            arguments={'generate_preview': 'True', 'row_count': '1000'})

        action_result_dto = _restclient(workspace).dataset.get_action_result(
            workspace.subscription_id,
            workspace.resource_group,
            workspace.name,
            dataset_id=_LEGACY_DATASET_ID,
            request=request_dto,
            custom_headers=_custom_headers)

        if action_result_dto.is_up_to_date is None:
            raise AzureMLException(action_result_dto.is_up_to_date_error)

        return not action_result_dto.is_up_to_date

    def __repr__(self):
        """Return string representation of ColumnProfile class.

        See https://docs.microsoft.com/en-us/python/api/azureml-dataprep/azureml.dataprep.columnprofile
        for more information on ColumnProfile.
        """
        return self._profile.__repr__()
