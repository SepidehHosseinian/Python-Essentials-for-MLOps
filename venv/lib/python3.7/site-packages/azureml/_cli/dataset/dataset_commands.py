# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json

from azureml._cli.dataset.dataset_subgroup import DatasetSubGroup
from azureml._cli.cli_command import command
from azureml._cli import argument
from azureml.exceptions import UserErrorException

from azureml.core.dataset import Dataset
from azureml.data._dataset_deprecation import deprecated
from azureml.data._dataset_persistence import generate_file_template, create_dataset_from_file


DATASET_NAME = argument.Argument(
    "dataset_name", "--name", "-n", required=False,
    help="Registration name of the dataset")
DATASET_VERSION = argument.Argument(
    "dataset_version", "--version", "-v", required=False, default='latest',
    help="Registration version of the dataset")
DATASET_ID = argument.Argument(
    "dataset_id", "--id", "-i", required=False,
    help="ID of the dataset saved to workspace")
DATASET_FILE = argument.Argument(
    "dataset_file", "--file", "-f", required=False,
    help="Specification file for dataset")
DATASET_SHOW_TEMPLATE = argument.Argument(
    "dataset_show_template", "--show-template", "", action="store_true", required=False,
    help="Show template of dataset specification file")
DATASET_SKIP_VALIDATE = argument.Argument(
    "dataset_skip_validate", "--skip-validation", "", action="store_true", required=False,
    help="Skip validation that ensures data can be loaded from the dataset before registration")
DATASET_REGISTERED_ID = argument.Argument(  # for legacy id system
    "dataset_id", "--id", "-i", required=False,
    help="Dataset ID (guid)")
DEPRECATE_BY_DATASET_ID = argument.Argument(
    "deprecate_by_dataset_id", "--deprecate-by-id", "-d", required=True,
    help="Dataset ID (guid) which is the intended replacement for this Dataset.")


def _check_python():
    # azureml-core is only installable with python>=3.5
    return True


def _dataset_to_printable(dataset):
    return json.loads(repr(dataset))


def _registration_to_printable(registration):
    return {
        'id': registration.saved_id,
        'name': registration.name,
        'version': registration.version,
        'description': registration.description,
        'tags': registration.tags
    }


@command(
    subgroup_type=DatasetSubGroup,
    command="list",
    short_description="List all datasets in the workspace")
def list_datasets_in_workspace(workspace=None, logger=None):
    registrations = Dataset.get_all(workspace).registrations
    return [_registration_to_printable(r) for r in registrations]


@command(
    subgroup_type=DatasetSubGroup,
    command="show",
    short_description="Get details of a dataset by its id or registration name",
    argument_list=[
        DATASET_NAME,
        DATASET_VERSION,
        DATASET_ID
    ])
def get_dataset(
        workspace=None,
        dataset_name=None,
        dataset_version=None,
        dataset_id=None,
        logger=None):
    if dataset_name is None and dataset_id is None:
        raise UserErrorException(
            'Argument {} or {} must be specified'
            .format(DATASET_NAME.long_form, DATASET_ID.long_form))
    if dataset_name is not None and dataset_id is not None:
        raise UserErrorException(
            'Arguments {} and {} cannot be specified at the same time'
            .format(DATASET_NAME.long_form, DATASET_ID.long_form))
    if dataset_version != DATASET_VERSION.default and dataset_name is None:
        raise UserErrorException(
            'Argument {} must be specified with {}'
            .format(DATASET_VERSION.long_form, DATASET_NAME.long_form))
    dataset_version = dataset_version or DATASET_VERSION.default
    if dataset_name is not None:
        dataset = Dataset.get_by_name(workspace, dataset_name, dataset_version)
    else:
        dataset = Dataset.get_by_id(workspace, dataset_id)
    return _dataset_to_printable(dataset)


@command(
    subgroup_type=DatasetSubGroup,
    command="register",
    short_description="Register a new dataset from the specified file",
    argument_list=[
        DATASET_SHOW_TEMPLATE,
        DATASET_FILE,
        DATASET_SKIP_VALIDATE
    ])
def register_dataset(
        workspace=None,
        dataset_show_template=None,
        dataset_file=None,
        dataset_skip_validate=None,
        logger=None):
    if _check_python() is False:
        raise UserErrorException('The dataset command subgroup is only supported with Python 3.5 or higher')
    if dataset_show_template:
        if dataset_file:
            raise UserErrorException(
                'Arguments {} and {} cannot be specified at the same time'
                .format(DATASET_SHOW_TEMPLATE.long_form, DATASET_FILE.long_form))
        return generate_file_template()
    if dataset_file is None:
        raise UserErrorException('Argument {} must be specified'.format(DATASET_FILE.long_form))
    dataset = create_dataset_from_file(workspace, dataset_file, not dataset_skip_validate, True)
    return _dataset_to_printable(dataset)


@command(
    subgroup_type=DatasetSubGroup,
    command="unregister",
    short_description="Unregister all versions under the specified registration name",
    argument_list=[
        DATASET_NAME
    ])
def unregister_dataset(
        workspace=None,
        dataset_name=None,
        logger=None):
    try:
        dataset = Dataset.get_by_name(workspace, dataset_name)
    except Exception:
        print('There is no dataset registered with name "{}".'.format(dataset_name))
        return
    dataset.unregister_all_versions()
    print('Successfully unregistered datasets with name "{}".'.format(dataset_name))


@deprecated(
    'The Dataset deprecation command.',
    'Command to deprecate an active dataset in a workspace by another dataset.')
@command(
    subgroup_type=DatasetSubGroup,
    command="deprecate",
    short_description="Deprecate an active dataset in a workspace by another dataset",
    argument_list=[
        DATASET_NAME,
        DATASET_REGISTERED_ID,
        DEPRECATE_BY_DATASET_ID
    ])
def deprecate_dataset(
        workspace=None,
        dataset_name=None,
        dataset_id=False,
        deprecate_by_dataset_id=None,
        logger=None):
    if _check_python() is False:
        raise UserErrorException('The dataset command subgroup is only supported with Python 3.5 or more')
    dataset = Dataset.get(workspace, dataset_name, dataset_id)
    dataset_state = dataset.state
    if dataset_state == 'deprecated':
        raise UserErrorException("Dataset '{}' ({}) is already deprecated".format(dataset.name, dataset.id))
    dataset.deprecate(deprecate_by_dataset_id)
    dataset = Dataset.get(workspace, name=dataset.name)
    if dataset.state == 'deprecated':
        logger.info("Dataset '{}' ({}) was deprecated successfully".format(dataset.name, dataset.id))
        return dataset._get_base_info_dict_show()
    else:
        logger.debug("dataset deprecate error. name: {} id: {} deprecate_by_id: {} state: {}".format(
            dataset.name, dataset.id, deprecate_by_dataset_id, dataset.state))
        raise Exception("Error, Dataset '{}' ({}) was not deprecated".format(dataset.name, dataset.id))


@deprecated(
    'The Dataset archive command',
    'Command to archive an active or deprecated dataset.')
@command(
    subgroup_type=DatasetSubGroup,
    command="archive",
    short_description="Archive an active or deprecated dataset",
    argument_list=[
        DATASET_NAME,
        DATASET_REGISTERED_ID
    ])
def archive_dataset(
        workspace=None,
        dataset_name=None,
        dataset_id=None,
        logger=None):
    if _check_python() is False:
        raise UserErrorException('The dataset command subgroup is only supported with Python 3.5 or more')
    dataset = Dataset.get(workspace, dataset_name, dataset_id)
    dataset_state = dataset.state
    if dataset_state == 'archived':
        raise UserErrorException("Dataset '{}' ({}) is already archived".format(dataset.name, dataset.id))
    dataset.archive()
    dataset = Dataset.get(workspace, name=dataset.name)
    if dataset.state == 'archived':
        logger.info("Dataset '{}' ({}) was archived successfully".format(dataset.name, dataset.id))
        return dataset._get_base_info_dict_show()
    else:
        logger.debug("dataset archive error. name: {} id: {} state: {}".format(
            dataset.name, dataset.id, dataset.state))
        raise Exception("Error, Dataset '{}' ({}) was not archived".format(dataset.name, dataset.id))


@deprecated(
    'The Dataset reactivate command',
    'Command to reactivate an archived or deprecated dataset.')
@command(
    subgroup_type=DatasetSubGroup,
    command="reactivate",
    short_description="Reactivate an archived or deprecated dataset",
    argument_list=[
        DATASET_NAME,
        DATASET_REGISTERED_ID
    ])
def reactivate_dataset(
        workspace=None,
        dataset_name=None,
        dataset_id=None,
        logger=None):
    if _check_python() is False:
        raise UserErrorException('The dataset command subgroup is only supported with Python 3.5 or more')
    dataset = Dataset.get(workspace, dataset_name, dataset_id)
    dataset_state = dataset.state
    if dataset_state == 'active':
        raise UserErrorException("Dataset '{}' ({}) is already active".format(dataset.name, dataset.id))
    dataset.reactivate()
    dataset = Dataset.get(workspace, name=dataset.name)
    if dataset.state == 'active':
        logger.info("Dataset '{}' ({}) was reactivated successfully".format(dataset.name, dataset.id))
        return dataset._get_base_info_dict_show()
    else:
        logger.debug("dataset reactivate error. name: {} id: {} state: {}".format(
            dataset.name, dataset.id, dataset.state))
        raise Exception("Error, Dataset '{}' ({}) was not reactivated".format(dataset.name, dataset.id))
