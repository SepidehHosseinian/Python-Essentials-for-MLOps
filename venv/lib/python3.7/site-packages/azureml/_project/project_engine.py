# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" projectsystem.py, A file for handling new project system logic."""

from __future__ import print_function

import traceback

import azureml._project.project_mapper as project_mapper
import azureml._project.project_manager as project_manager
import azureml._project.project_info as ProjectInfo
from azureml.exceptions import ProjectSystemException

# JSON File Keys
PROJECT_ID = "Id"
PROJECT_PATH = "Path"
EXPERIMENT_NAME = "RunHistoryName"
USER_EMAIL = "UserEmail"
USER_NAME = "Username"
BEHALF = "BehalfOfMicrosoft"
ACCOUNT_NAME = "AccountName"


def _raise_request_error(response, action="calling backend service"):
    if response.status_code >= 400:
        from azureml._base_sdk_common.common import get_http_exception_response_string
        raise ProjectSystemException(get_http_exception_response_string(response))


class ProjectEngineClient(object):
    def __init__(self, auth):
        """

        :param auth: auth object
        :type auth: azureml.core.authentication.AbstractAuthentication
        """
        self._auth = auth
        # Repo related information is none for now
        self._headers = auth.get_authentication_header()

    def attach_project(self, project_id, project_path, project_arm_scope, compute_target_dict):
        """
        Attaches a local folder, specified by project_path, as an
        azureml project.
        One thing to note is that in this we don't create or delete any project
        directory, otherwise we may end up deleting a users C drive in the worst case.
        """
        try:
            project_manager.attach_project(project_id, project_path, project_arm_scope, compute_target_dict)

            return {
                PROJECT_ID: project_id,
                PROJECT_PATH: project_path
            }

        except Exception:
            raise ProjectSystemException(traceback.format_exc())

    @staticmethod
    def get_project_scope_by_path(project_path):
        try:
            project_info = ProjectInfo.get(project_path)
            if not project_info:
                return None

            return project_info.Scope

        except Exception:
            raise ProjectSystemException(traceback.format_exc())

    def get_local_projects(self):
        try:
            return project_mapper.get_project_id_to_path_map()

        except Exception:
            raise ProjectSystemException(traceback.format_exc())
