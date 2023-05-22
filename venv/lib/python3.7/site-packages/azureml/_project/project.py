# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os


# Az CLI converts the keys to camelCase and our tests assume that behavior,
# so converting for the SDK too.
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.exceptions import UserErrorException
from azureml.core.runconfig import RunConfiguration

from azureml._base_sdk_common.common import check_valid_resource_name


from azureml._project import _commands
from azureml._project import _compute_target_commands
from azureml._restclient.snapshots_client import SnapshotsClient


class Project(object):
    """
    The project class for local on-disk projects.
    """

    def __init__(self, directory=".", experiment=None, auth=None, _disable_service_check=False):
        """
        Creates the project object using the local project path.
        :param directory: Project path.
        :type directory: str
        :param experiment:
        :type experiment: azureml.core.Experiment
        :param auth: An authentication object of a subclass of azureml.core.authentication.AbstractAuthentication
        :type auth: azureml.core.authentication.AbstractAuthentication
        :return:
        """
        from azureml.core.experiment import Experiment
        if not directory:
            directory = "."
        if experiment:
            self._workspace = experiment.workspace
            self.directory = directory
            self._project_path = os.path.abspath(directory)
            self._experiment = experiment
            self._snapshots_client = SnapshotsClient(self._workspace.service_context)

        else:
            if not auth:
                auth = InteractiveLoginAuthentication()

            self._project_path = os.path.abspath(directory)

            info_dict = _commands.get_project_info(auth, self._project_path)

            from azureml.core.workspace import Workspace
            self._workspace = Workspace(info_dict[_commands.SUBSCRIPTION_KEY],
                                        info_dict[_commands.RESOURCE_GROUP_KEY],
                                        info_dict[_commands.WORKSPACE_KEY],
                                        auth, _disable_service_check=_disable_service_check)
            self._experiment = Experiment(self._workspace, info_dict[_commands.PROJECT_KEY])
            self._snapshots_client = SnapshotsClient(self._workspace.service_context)

    @property
    def workspace(self):
        """
        :return: Returns the workspace object corresponding to this project.
        :rtype: azureml.core.workspace.Workspace
        """
        return self._workspace

    @property
    def history(self):
        """
        :return: Returns the run history corresponding to this project.
        :rtype: azureml.core.experiment.Experiment
        """
        return self._experiment

    @property
    def experiment(self):
        """
        :return: Returns the experiment attached to this project.
        :rtype: azureml.core.experiment.Experiment
        """
        return self._experiment

    @staticmethod
    def attach(workspace_object, experiment_name, directory="."):
        """
        Attaches the project, specified by directory, as an azureml project to
        the specified workspace and run history.
        If the path specified by directory doesn't exist then we create those directories.
        :param workspace_object: The workspace object.
        :type workspace_object: azureml.core.workspace.Workspace
        :param experiment_name: The experiment name.
        :type experiment_name: str
        :param directory: The directory path.
        :type directory: str
        :return: The project object.
        :rtype: azureml.core.project.Project
        """
        if not directory:
            directory = "."

        check_valid_resource_name(experiment_name, "Experiment")

        _commands.attach_project(workspace_object, experiment_name,
                                 project_path=directory)

        # The project is created inside directory with name project_name.
        # So the project directory is os.path.join(directory, project_name)
        project = Project(auth=workspace_object._auth_object, directory=directory)

        return project

    @property
    def project_directory(self):
        """
        Returns the local on-disk project path.
        :return:
        """
        return self._project_path

    def detach(self):
        """
        Detaches the current project from being an azureml project.
        Throws an exception if detach fails.
        :return: None
        """
        # TODO: Nice errors for the detached projects object reuse.
        _commands.detach_project(self.project_directory)

    def get_details(self):
        """
        Returns the details of the current project.
        :return:
        :rtype: dict
        """
        return self._serialize_to_dict()

    @property
    def legacy_compute_targets(self):
        """
        Returns legacy compute targets as a dictionary.
        Key is the compute target name, value is the compute target type
        :return:
        :rtype: dict
        """
        return _compute_target_commands.get_all_compute_target_objects(self)

    # To be made public in future.
    def _take_snapshot(self):
        """
        Take a snapshot of the project.
        :return: SnapshotId
        """
        return self._snapshots_client.create_snapshot(self.project_directory)

    # To be made public in future.
    def _snapshot_restore(self, snapshot_id, path=None):
        """
        Restores a project to a snapshot, specified by the snapshot_id.
        :param snapshot_id: The snapshot id to restore to.
        :type snapshot_id: str
        :param path: The path where the project should be restored.
        :type path: str
        :return: The path.
        :rtype: str
        """
        return self._snapshots_client.restore_snapshot(snapshot_id, path)

    def _get_run_config_object(self, run_config):
        if isinstance(run_config, str):
            # If it is a string then we don't need to create a copy.
            return RunConfiguration.load(self.project_directory, run_config)
        elif isinstance(run_config, RunConfiguration):
            # TODO: Deep copy of project and auth object too.
            import copy
            return copy.deepcopy(run_config)
        else:
            raise UserErrorException("Unsupported runconfig type {}. run_config can be of str or "
                                     "azureml.core.runconfig.RunConfiguration type.".format(type(run_config)))

    def _serialize_to_dict(self):
        """
        Serializes the Project object details into a dictionary.
        :return:
        :rtype: dict
        """
        output_dict = self.history._serialize_to_dict()
        output_dict["Project path"] = self.project_directory
        return output_dict
