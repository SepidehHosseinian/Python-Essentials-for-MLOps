# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Contains configuration to generate statistics summary of datasets in Azure Machine Learning.

Functionality in this module includes methods for submitting local or remote profile run and
visualizing the result of the submitted profile run.
"""

import os
import sys
from shutil import copyfile
from azureml.exceptions import UserErrorException
from backports import tempfile
import uuid
from azureml.core import ComputeTarget, RunConfiguration, \
    ScriptRunConfig, Experiment, get_run
from azureml.core._experiment_method import experiment_method
from azureml.data import TabularDataset
from azureml.data._dataset_rest_helper import _restclient, _custom_headers
from azureml.data.constants import _LOCAL_COMPUTE
from azureml.data.dataset_profile_run import DatasetProfileRun
from azureml._base_sdk_common._docstring_wrapper import experimental


def _submit_profile(dataset_profile_config_object, workspace, experiment_name):
    """Start Profile execution with the given config on the given workspace.

    :param dataset_profile_config_object:
    :param workspace:
    :param experiment_name:
    :param kwargs:
    :return:
    """
    dataset = dataset_profile_config_object._dataset
    compute_target = dataset_profile_config_object._compute_target
    datastore_name = dataset_profile_config_object._datastore_name

    if isinstance(compute_target, ComputeTarget):
        compute_target = compute_target.name
    else:
        compute_target = compute_target
    run_id = 'dataset_' + str(uuid.uuid4())
    saved_dataset_id = dataset._ensure_saved(workspace)
    action_dto = _restclient(workspace).dataset.generate_profile_with_preview(
        workspace.subscription_id,
        workspace.resource_group,
        workspace.name,
        id=saved_dataset_id,
        compute_target=compute_target,
        experiment_name=experiment_name,
        run_id=run_id,
        datastore_name=datastore_name,
        custom_headers=_custom_headers)

    if dataset_profile_config_object._compute_target == _LOCAL_COMPUTE:
        with tempfile.TemporaryDirectory() as temp_dir:
            script = os.path.join(temp_dir, 'profile_run_script.py')
            copyfile(os.path.join(os.path.dirname(__file__), '_profile_run_script.py'), script)
            run_local = RunConfiguration()
            run_local.environment.python.user_managed_dependencies = True
            run_local.environment.python.interpreter_path = sys.executable
            script_config = ScriptRunConfig(
                source_directory=temp_dir,
                script="profile_run_script.py",
                arguments=[action_dto.dataset_id, action_dto.action_id, saved_dataset_id],
                run_config=run_local)
            experiment = Experiment(workspace, experiment_name)
            experiment.submit(script_config, run_id=run_id)
    else:
        experiment = Experiment(workspace, action_dto.experiment_name)
        run_id = action_dto.run_id
    run = get_run(experiment, run_id)
    return DatasetProfileRun(workspace, dataset, run)


@experimental
class DatasetProfileRunConfig(object):
    """Represents configuration for submitting a dataset profile run in Azure Machine Learning.

    This configuration object contains and persists the parameters for configuring the experiment run,
    as well as the data to be used at run time.

    .. remarks::

        The following code shows a basic example of creating an DatasetProfileRunConfig object and submitting an
        experiment for profile computation on 'local':

        .. code-block:: python

            workspace = Workspace.from_config()
            dataset = Dataset.get_by_name(workspace, name='test_dataset')
            dprc = DatasetProfileRunConfig(dataset=dataset, compute_target='local')
            exp = Experiment(workspace, "profile_experiment_run")
            profile_run = exp.submit(dprc)
            profile_run.run.wait_for_completion(raise_on_error=True, wait_post_processing=True)
            profile = profile_run.get_profile()

    :param dataset: The tabular dataset which is target of this profile run..
        See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset
        for more information on tabular datasets.
    :type dataset: azureml.data.TabularDataset
    :param compute_target: The Azure Machine Learning compute target to run the
        Automated Machine Learning experiment on. Specify 'local' to use local compute.
        Default is 'local'.
        See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.computetarget
        for more information on compute targets.
    :type compute_target: typing.Union[azureml.core.compute.ComputeTarget, str]
    """

    @experiment_method(submit_function=_submit_profile)
    def __init__(self, dataset, compute_target='local', datastore_name=None):
        """Create a DatasetProfileRunConfig object.

        :param dataset: The tabular dataset which is target of this profile run.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.tabulardataset
            for more information on tabular datasets.
        :type dataset: azureml.data.TabularDataset
        :param compute_target: The Azure Machine Learning compute target to run the
            Automated Machine Learning experiment on. Specify 'local' to use local compute.
            Default is 'local'.
            See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.computetarget
            for more information on compute targets.
        :type compute_target: typing.Union[azureml.core.compute.ComputeTarget, str]
        :param datastore_name: the name of datastore to store the profile cache
        :type datastore_name: str
        """
        self._validate_inputs(dataset, compute_target)
        self._dataset = dataset
        self._compute_target = compute_target
        self._datastore_name = datastore_name

    @staticmethod
    def _validate_inputs(dataset, compute_target):

        if not isinstance(dataset, TabularDataset):
            raise UserErrorException('Invalid type. dataset should be of type '
                                     'azureml.data.tabular_dataset.TabularDataset but was found to be '
                                     'of type {0}.'.format(type(dataset)))

        if not (isinstance(compute_target, ComputeTarget) or isinstance(compute_target, str)):
            raise UserErrorException('Invalid type. compute_target should be either of type ComputeTarget or '
                                     'string but was found to be of type {0}.'.format(type(compute_target)))
