# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import azureml._project.project_info as project_info
from azureml._project.mappings import ProjectMappings, RepoKeys


def get_project_id_to_path_map():
    """
    Returns a mapping of project_id to it's local path

    :rtype: {}
    """
    project_mapping = ProjectMappings()
    mappings = project_mapping.get_values()

    projects = {}
    for project_path in mappings:
        info = project_info.get(project_path)
        if info and info.Id:
            projects[info.Id] = project_path
        else:
            project_mapping.delete(project_path)

    return projects


def get_repo_key_from_path(path):
    """
    Get the repo key for specified project

    :type path: str

    :rtype: str
    """
    info = project_info.get(path)
    if info and info.Id:
        return RepoKeys().get(info.Id)
    return None


def add_project(project_id, path, scope, repo_key=None):
    """
    Add a new project to mapping

    :type project_id: str
    :type path: str
    :type scope: str
    :type repo_key: str

    :rtype: None
    """
    project_info.add(project_id, scope, path)
    if repo_key:
        RepoKeys().add(project_id, repo_key)

    ProjectMappings().add(path)


def remove_project(path):
    """
    Removed specified project from mapping

    :type path: str

    :rtype: None
    """
    ProjectMappings().delete(path)
    info = project_info.get(path)
    if info and info.Id:
        RepoKeys().delete(info.Id)

    project_info.delete(path)
