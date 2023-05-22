# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import shutil
import json

from azureml._project.ignore_file import AmlIgnoreFile
import azureml._project.file_utilities as file_utilities
import azureml._project.project_info as project_info
import azureml._project.project_mapper as project_mapper

from azureml._base_sdk_common import __version__ as package_version

_default_git_folder_name = ".git"
_asset_folder_name = "assets"
_base_project_contents_folder_name = "base_project_files"
_conda_dependencies_file_name = "conda_dependencies.yml"

_history_branch_name = "AzureMLHistory"

_link_repo_commit_message = "link"
_create_project_commit_message = "Initial commit"
_run_history_push_commit_message = "Run history"


def _current_index():
    requirements_index = None
    index_file_path = os.path.join(os.path.dirname(__file__), "index_location.txt")
    with open(index_file_path, "r") as file:
        prerelease_index = file.read().strip()
        if prerelease_index:
            requirements_index = prerelease_index
    return requirements_index


def _sdk_scope():
    scope = []
    scope_file_path = os.path.join(os.path.dirname(__file__), "azureml_sdk_scope.txt")
    if os.path.exists(scope_file_path):
        with open(scope_file_path, "r") as file:
            scope = [line.strip() for line in file.readlines()]
    return scope


def _base_images_current_tags():
    images = {}
    current_base_images_file = os.path.join(os.path.dirname(__file__), "azureml_base_images.json")
    if os.path.exists(current_base_images_file):
        try:
            with open(current_base_images_file) as file:
                images = json.loads(file.read())
        except Exception:
            pass
    return images


def _get_tagged_image(image_name, default_tag=None):
    """
    Return tagged image from azureml_base_images.json, pin to default_tag if missing, else as is
    """
    images = _base_images_current_tags()
    tag = images.get(image_name, None)
    if tag:
        return image_name + ":" + tag
    else:
        return image_name + ((":" + default_tag) if default_tag else "")


def _update_requirements_binding(repo_path, config_dir_to_use):
    # These should remain None for the local development scenario.
    requirements_version = None

    # Set the package version from the __version__ if it's not local development default.
    if not package_version.endswith("+dev"):
        requirements_version = package_version

    requirements_index = _current_index()

    default_index = "https://azuremlsdktestpypi.azureedge.net/sdk-release/Preview/E7501C02541B433786111FE8E140CAA1"
    conda_dependencies_path = os.path.join(repo_path, config_dir_to_use, _conda_dependencies_file_name)

    lines = []
    with open(conda_dependencies_path, "r") as infile:
        for line in infile:
            if requirements_version:
                line = line.replace("azureml-defaults", "azureml-defaults==" + requirements_version)
            if requirements_index:
                line = line.replace(default_index, requirements_index)

            lines.append(line)
    with open(conda_dependencies_path, 'w') as outfile:
        for line in lines:
            outfile.write(line)


def attach_project(project_id, project_path, scope, compute_target_dict):
    """
    Attaches a local folder specified by project_path as a project.

    :type project_id:  str
    :type project_path:  str
    :type scope:  str
    :rtype: None
    """
    from azureml._base_sdk_common.common import get_run_config_dir_name
    is_existing_dir = os.path.isdir(project_path)
    if not is_existing_dir:
        # We creating all intermediate dirs too.
        os.makedirs(os.path.abspath(project_path))

    # check path is a full, rooted path
    if not os.path.isabs(project_path):
        raise ValueError("Selected directory is invalid")

    # For backcompat case, where if path already has aml_config then we just use that, instead of
    # creating .azureml
    confing_dir_name_to_use = get_run_config_dir_name(project_path)

    # check if path is already a project
    original_project_info = project_info.get(project_path, no_recursive_check=True)

    _create_metadata_folders(project_path, confing_dir_name_to_use)

    # Only copying when repo_path is not already a project.
    if not original_project_info:
        _copy_default_files(os.path.join(project_path, confing_dir_name_to_use),
                            _base_project_contents_folder_name)
        _update_requirements_binding(project_path, confing_dir_name_to_use)

        # Creates local and docker runconfigs.
        _create_default_run_configs(project_path, compute_target_dict)

    # Overwriting if project.json already exists.
    project_mapper.add_project(project_id, project_path, scope)


def delete_project(path):
    """
    Removes project from mapping. Does not delete entire project from disk.

    :type path: str

    :rtype: None
    """
    project_mapper.remove_project(path)


def _copy_default_files(path, default_fileset):
    """
    Copy default files to folder

    :type path: str
    :rtype: None
    """
    this_dir, this_filename = os.path.split(__file__)
    default_files_path = os.path.join(this_dir, default_fileset)

    if not os.path.exists(path):
        os.mkdir(path)
    for filename in os.listdir(default_files_path):
        orig_path = os.path.join(default_files_path, filename)
        new_path = os.path.join(path, filename)
        if os.path.isdir(orig_path):
            shutil.copytree(orig_path, new_path)
        else:
            if not os.path.exists(new_path):
                shutil.copy(orig_path, new_path)


def _create_metadata_folders(path, confing_dir_name_to_use):
    """
    Create metadata files and folders
    :type path: str
    :rtype: None
    """
    file_utilities.create_directory(os.path.join(path, confing_dir_name_to_use))

    aml_ignore = AmlIgnoreFile(path)
    aml_ignore.create_if_not_exists()


def _ensure_directory_is_valid(path):
    """
    Validate the directory

    :type path: str

    :rtype: None
    """
    # check path is a full, rooted path
    if not os.path.isabs(path):
        raise ValueError("Selected directory is invalid")

    # check if path is already a project
    if project_info.get(path):
        raise ValueError("Directory must not be an existing project")


def empty_function():
    return


def _create_default_run_configs(project_directory, compute_target_dict):
    """
    Creates a local.runconfig and docker.runconfig for a project.
    :return: None
    """
    from azureml.core.runconfig import RunConfiguration
    # Mocking a project object, as RunConfiguration requires a Project object, but only requires
    # project_directory field.
    project_object = empty_function
    project_object.project_directory = project_directory

    # Creating a local runconfig.
    local_run_config = RunConfiguration()
    local_run_config.save(name="local", path=project_directory)

    # Creating a docker runconfig.
    docker_run_config = RunConfiguration()
    docker_run_config.environment.docker.enabled = True
    docker_run_config.save(name="docker", path=project_directory)

    for compute_target_name, compute_target in compute_target_dict.items():
        # Creating a compute runconfig.
        compute_config = RunConfiguration()
        if compute_target.type == 'HDInsight':
            compute_config.framework = "PySpark"
        else:
            compute_config.framework = "Python"
            compute_config.environment.docker.enabled = True
        compute_config.target = compute_target_name
        compute_config.save(name=compute_target_name, path=project_directory)
