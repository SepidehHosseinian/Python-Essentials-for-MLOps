# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------

"""Access run artifacts client"""
import requests

from msrest.exceptions import HttpOperationError

from azureml._file_utils import download_file
from azureml.exceptions import UserErrorException

from .experiment_client import ExperimentClient

SUPPORTED_NUM_EMPTY_ARTIFACTS = 50


# This is only a subset of the functionality that the RunArtifacts Facade in RunHistory allows
# This subset is chosen based on the CLI's requirements
class RunArtifactsClient(ExperimentClient):
    """
    Run History Artifact Facade APIs

    :param host: The base path for the server to call.
    :type host: str
    :param auth: Authentication for the client
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
                 **kwargs):
        super(RunArtifactsClient, self).__init__(service_context, experiment_name, **kwargs)

        batch_size = SUPPORTED_NUM_EMPTY_ARTIFACTS

        self.session = requests.session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_maxsize=batch_size))

    def get_artifact_by_container(self, run_id):
        """
        Get artifact names of a run by their container
        :param run_id:  (required)
        :type run_id: str
        :return: a generator of ~_restclient.models.ArtifactDto
        """
        return self._execute_with_experiment_arguments(self._client.run_artifact.list_in_container,
                                                       run_id=run_id,
                                                       is_paginated=True)

    def get_artifact_content(self, run_id, path):
        """
        Get artifact names of a run by their container
        :param run_id:  (required)
        :type run_id: str
        :param path: (required)
        :type path: str
        :return: a generator of ~_restclient.models.ArtifactDto
        """
        return self._execute_with_experiment_arguments(self._client.run_artifact.get_content_information,
                                                       run_id=run_id,
                                                       path=path)

    def download_artifact(self, run_id, path, output_file_path):
        """download artifact"""
        try:
            content_info = self._execute_with_experiment_arguments(self._client.run_artifact.get_content_information,
                                                                   run_id, path)
            if not content_info:
                raise UserErrorException("Cannot find the artifact '{0}' in container '{1}'".format(path, run_id))
            uri = content_info.content_uri
        except HttpOperationError as operation_error:
            if operation_error.response.status_code == 404:
                existing_files = self.get_file_paths(run_id)
                raise UserErrorException("File with path {0} was not found,\n"
                                         "available files include: "
                                         "{1}.".format(path, ",".join(existing_files)))
            else:
                raise
        download_file(uri, output_file_path)

    def get_file_paths(self, run_id):
        """list artifact info"""
        artifacts = self._execute_with_experiment_arguments(self._client.run_artifact.list_in_container,
                                                            run_id=run_id,
                                                            is_paginated=True)

        return map(lambda artifact_dto: artifact_dto.path, artifacts)
