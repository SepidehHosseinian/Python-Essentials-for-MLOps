# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Class for handling and monitoring dataset profile run associated with an Experiment object and individual run id."""

from azureml.core.run import Run
from azureml.data.constants import _LEGACY_DATASET_ID, _PROFILE_RUN_ACTION_ID
from azureml.data._dataset_rest_helper import _restclient, _custom_headers
from azureml.data._dataprep_helper import dataprep
from azureml._restclient.artifacts_client import ArtifactsClient


class DatasetProfileRun(Run):
    """An experiment run class to handle and monitor dataset profile run associated with an Experiment object and
    individual run id.
    """

    def __init__(self, experiment, run_id, action_id=None, **kwargs):
        """Class DatasetProfileRun constructor."""
        super().__init__(experiment, run_id, **kwargs)
        if _PROFILE_RUN_ACTION_ID in self.properties:
            self._action_id = self.properties[_PROFILE_RUN_ACTION_ID]
        elif action_id is not None:
            self._action_id = action_id
            self.add_properties({_PROFILE_RUN_ACTION_ID: action_id})
        else:
            raise ValueError(
                'Run "{}" for experiment "{}" is not a DatasetProfileRun.'.format(experiment.name, run_id))
        self._workspace = experiment.workspace

    @property
    def profile(self):
        """Retrieve the data profile from result of this run, meanwhile checking if it matches current data.

        :return: A tuple of values. The first value is the data profile result from the completed run. The second
            value is a flag indicating whether the profile matches current data:
                - if the data source change cannot be detected, the flag will be None;
                - if the data source was changed after submitting the profile run, the flag will be False;
                - otherwise, the profile matches current data, and the flag will be True.
        :rtype: (azureml.dataprep.DataProfile, bool)
        """
        if self.status != 'Completed':
            return (None, None)
        action_dto = _restclient(self._workspace).dataset.get_action_by_id(
            self._workspace.subscription_id,
            self._workspace.resource_group,
            self._workspace.name,
            dataset_id=_LEGACY_DATASET_ID,
            action_id=self._action_id,
            _custom_headers=_custom_headers)
        return _profile_from_action(self._workspace, action_dto)


def _profile_from_action(workspace, result):
    result_artifact_ids = result.result_artifact_ids
    if result_artifact_ids is None or len(result_artifact_ids) == 0:
        return (None, None)
    result_artifact = result_artifact_ids[0]
    content = ArtifactsClient(workspace.service_context).download_artifact_contents_to_string(
        *result_artifact.split("/", 2))
    try:
        profile = dataprep().DataProfile._from_json(content)
    except Exception:
        raise RuntimeError('Profile result is corrupted.')
    if hasattr(result, 'is_up_to_date_error') and result.is_up_to_date_error:
        raise RuntimeError(result.is_up_to_date_error)
    if hasattr(result, 'is_up_to_date'):
        return (profile, result.is_up_to_date)
    return (profile, None)
