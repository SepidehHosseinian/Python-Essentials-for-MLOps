# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os
import shutil
from azureml.exceptions import UserErrorException

import azureml._project.file_utilities as file_utilities

from azureml._base_sdk_common.common import AML_CONFIG_DIR, AZUREML_DIR, CONFIG_FILENAME, \
    LEGACY_PROJECT_FILENAME


def add(project_id, scope, project_path, is_config_file_path=False):
    """
    Creates project info file

    :type project_id: str
    :type scope: str
    :type project_path: str

    :rtype: None
    """
    if is_config_file_path:
        project_file_path = project_path
    else:
        from azureml._base_sdk_common.common import get_run_config_dir_name, get_config_file_name
        config_dir_name = get_run_config_dir_name(project_path)
        config_directory = os.path.join(project_path, config_dir_name)
        file_utilities.create_directory(config_directory, True)
        config_file_name = get_config_file_name(config_directory)
        project_file_path = os.path.join(config_directory, config_file_name)
    # We overwriting if project.json exists.
    with open(project_file_path, "w") as fo:
        info = ProjectInfo(project_id, scope)
        fo.write(json.dumps(info.__dict__))


def get_workspace_info(found_path):
    with open(found_path, 'r') as config_file:
        config = json.load(config_file)

    # Checking the keys in the config.json file to check for required parameters.
    scope = config.get('Scope')
    if not scope:
        if not all([k in config.keys() for k in ('subscription_id', 'resource_group', 'workspace_name')]):
            raise UserErrorException('The config file found in: {} does not seem to contain the required '
                                     'parameters. Please make sure it contains your subscription_id, '
                                     'resource_group and workspace_name.'.format(found_path))
        # User provided ARM parameters take precedence over values from config.json
        subscription_id_from_config = config['subscription_id']
        resource_group_from_config = config['resource_group']
        workspace_name_from_config = config['workspace_name']
    else:
        pieces = scope.split('/')
        # User provided ARM parameters take precedence over values from config.json
        subscription_id_from_config = pieces[2]
        resource_group_from_config = pieces[4]
        workspace_name_from_config = pieces[8]
    return (subscription_id_from_config, resource_group_from_config, workspace_name_from_config)


def get(project_path, no_recursive_check=False):
    """
    Get ProjectInfo for specified project

    :type project_path: str
    :param no_recursive_check:
    :type no_recursive_check: bool
    :rtype: ProjectInfo
    """
    while True:
        # Although, we iterate but only one directory exists at a time.
        for config_path in [AML_CONFIG_DIR, AZUREML_DIR]:
            for files_to_look in [LEGACY_PROJECT_FILENAME, CONFIG_FILENAME]:
                config_file_path = os.path.join(project_path, config_path, files_to_look)
                if os.path.exists(config_file_path):
                    with open(config_file_path) as info_json:
                        config_json = json.load(info_json)
                        # If Scope is not there, then this is an old workspace config.json config file
                        # We ignore this file and try to get the project.json
                        # If user is invoking this function then they should definitely have
                        # a config.json or project.json that has Scope in it.
                        if config_json.get("Scope"):
                            project_info = ProjectInfo(config_json.get("Id"), config_json.get("Scope"))
                            return project_info

        parent_dir = os.path.dirname(project_path)
        if project_path == parent_dir:
            break
        else:
            project_path = parent_dir

        if no_recursive_check:
            return None
    return None


def delete_project_json(project_path):
    """
    Deletes the project.json from the project folder specified by project_path.
    :return: None, throws an exception if deletion fails.
    """
    for config_path in [AML_CONFIG_DIR, AZUREML_DIR]:
        legacy_info_path = os.path.join(project_path, config_path, LEGACY_PROJECT_FILENAME)
        info_path = os.path.join(project_path, config_path, CONFIG_FILENAME)
        if os.path.exists(legacy_info_path):
            os.remove(legacy_info_path)
        if os.path.exists(info_path):
            os.remove(info_path)


def delete(project_path):
    """
    Deletes the metadata folder containing project info

    :type project_path: str

    :rtype: None
    """
    config_directory = os.path.join(project_path, AZUREML_DIR)
    if os.path.isdir(config_directory):
        shutil.rmtree(config_directory)
    legacy_config_directory = os.path.join(project_path, AML_CONFIG_DIR)
    if os.path.isdir(legacy_config_directory):
        shutil.rmtree(legacy_config_directory)


class ProjectInfo(object):
    def __init__(self, project_id, scope):
        """
        :type project_id: str
        :type scope: str
        """
        # Uppercase to work with existing JSON files
        self.Id = project_id
        self.Scope = scope
