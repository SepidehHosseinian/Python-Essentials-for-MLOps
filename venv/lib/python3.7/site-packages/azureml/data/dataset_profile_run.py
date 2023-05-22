# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Contains configuration for monitoring dataset profile run in Azure Machine Learning.

Functionality in this module includes handling and monitoring dataset profile run associated
with an experiment object and individual run id.
"""

from azureml.core import Run
from azureml._base_sdk_common._docstring_wrapper import experimental


@experimental
class DatasetProfileRun(Run):
    """An experiment run class to handle and monitor dataset profile run.

    :param workspace: Workspace which the profile run belongs to.
        See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace
        for more information on workspaces.
    :type workspace: azureml.core.workspace.Workspace
    :param dataset: The tabular dataset which is target of this profile run.
        See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset
        for more information on tabular datasets.
    :type dataset: azureml.data.TabularDataset
    :param run: The submitted experiment run for this profile run.
        See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)
        for more information on run.
    :type run: azureml.core.Run
    """

    def __init__(self, workspace, dataset, run):
        """Create a DatasetProfileRun object.

        :param workspace: Workspace which the profile run belongs to.
            Required if dataset is not associated to a workspace.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace
            for more information on workspaces.
        :type workspace: azureml.core.workspace.Workspace
        :param dataset: The tabular dataset which is target of this profile run.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset
            for more information on tabular datasets.
        :type dataset: azureml.data.TabularDataset
        :param run: The submitted experiment run for this profile run.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)
            for more information on run.
        :type run: azureml.core.Run
        """
        super().__init__(run.experiment, run.id)
        self.workspace = workspace
        self.dataset = dataset
        self.run = run

    @experimental
    def get_profile(self):
        """Retrieve data profile from the result of this run.

        :return: Profile result from the latest profile run of type DatasetProfile
        :rtype: azureml.data.dataset_profile.DatasetProfile
        """
        return self.dataset.get_profile()
