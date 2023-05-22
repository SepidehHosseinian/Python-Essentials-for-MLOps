# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality to load datasets from definition files."""

import os
import json
import re
from collections import OrderedDict
from uuid import uuid4

from azureml.core import Datastore
from azureml.data import TabularDataset, FileDataset
from azureml.data.dataset_factory import TabularDatasetFactory, FileDatasetFactory, _validate_and_normalize_path
from azureml.data.constants import _DATASET_PROP_TIMESTAMP_FINE, _DATASET_PROP_TIMESTAMP_COARSE
from azureml.data._dataprep_helper import is_dataprep_installed, get_dataprep_missing_message


_PROP_SCHEMA_VERSION = 'schemaVersion'
_LATEST_SCHEMA_VERSION = 1


def generate_file_template():
    """Used by CLI to get a JSON template for dataset."""
    return _DatasetPersistenceV1.generate_template()  # Update when _LATEST_SCHEMA_VERSION changes


def create_dataset_from_file(workspace, def_path, validate, register):
    """Used by CLI to create dataset instance from definition file.

    :param workspace: The workspace to create the dataset.
    :type workspace: azureml.core.Workspace
    :param def_path: Path to the definition file.
    :type def_path: str
    :param validate: Whether to validate if data can be loaded from the dataset.
    :type validate: bool
    :param register: whether to register the dataset.
    :type register: bool
    """
    if not os.path.isfile(def_path):
        raise FileNotFoundError(def_path)

    if validate and not is_dataprep_installed():
        _raise_dataprep_missing_error('Cannot perform validation', 'setting --skip-validation')

    with open(def_path, 'r') as definition_file:
        try:
            definition = json.load(definition_file)
        except json.decoder.JSONDecodeError as e:
            raise json.decoder.JSONDecodeError('Malformatted JSON file: {}'.format(def_path), e.doc, e.pos)
        error_utility = _ErrorUtility(os.path.abspath(def_path))
        json_utility = _JsonUtility(error_utility)
        schema_version = json_utility.try_get_value(
            definition, _PROP_SCHEMA_VERSION, None,
            lambda v: isinstance(v, int) and v >= 1 and v <= _LATEST_SCHEMA_VERSION,
            '{} must be between {} and {}'.format(_PROP_SCHEMA_VERSION, 1, _LATEST_SCHEMA_VERSION)
        )
        if schema_version == 1:
            persistence = _DatasetPersistenceV1(workspace, definition, json_utility, error_utility)
        else:
            error_utility.raise_error('Unsupported schema version: {}.'.format(schema_version))

        return persistence.create_and_register_dataset(validate)


class _DatasetPersistenceV1:
    _valid_dataset_types = {'File', 'Tabular'}
    _valid_source_types = {'delimited_files', 'json_lines_files', 'parquet_files', 'sql_query'}

    _prop_dataset_type = 'datasetType'

    _prop_parameters = 'parameters'
    _prop_source_type = 'sourceType'
    _prop_path = 'path'
    _prop_query = 'query'
    _prop_datastore_name = 'datastoreName'
    _prop_relative_path = 'relativePath'
    _prop_partition_format = 'partitionFormat'
    _prop_include_path = 'includePath'
    _prop_infer_column_types = 'inferColumnTypes'
    _prop_separator = 'separator'
    _prop_header = 'header'

    _prop_time_series = 'timeSeries'
    _prop_fine_grain_timestamp = 'fineGrainTimestamp'
    _prop_coarse_grain_timestamp = 'coarseGrainTimestamp'

    _prop_registration = 'registration'
    _prop_name = 'name'
    _prop_create_new_version = 'createNewVersion'
    _prop_description = 'description'
    _prop_tags = 'tags'

    _default_include_path = False
    _default_infer_column_types = False
    _default_separator = ','
    _default_header = True

    @classmethod
    def generate_template(cls):
        template = OrderedDict([
            (_PROP_SCHEMA_VERSION, 1),
            (cls._prop_dataset_type, '|'.join(cls._valid_dataset_types)),
            (cls._prop_parameters, OrderedDict([
                (cls._prop_query, OrderedDict([
                    (cls._prop_datastore_name, 'sample-sql-datastore'),
                    (cls._prop_query, 'SELECT * FROM SAMPLE')
                ])),
                (cls._prop_source_type, '|'.join(cls._valid_source_types)),
                (cls._prop_path, [
                    'https://url/datafile',
                    OrderedDict([
                        (cls._prop_datastore_name, 'sample-blob-datastore'),
                        (cls._prop_relative_path, 'sample-data/datafile')
                    ])
                ]),
                (cls._prop_include_path, cls._default_include_path),
                (cls._prop_partition_format, '/{Country}/{PartitionDate:yyyy/MM/dd}/data.csv'),
                (cls._prop_infer_column_types, cls._default_infer_column_types),
                (cls._prop_separator, cls._default_separator),
                (cls._prop_header, cls._default_header)
            ])),
            (cls._prop_time_series, OrderedDict([
                (cls._prop_fine_grain_timestamp, 'column1'),
                (cls._prop_coarse_grain_timestamp, 'column2')
            ])),
            (cls._prop_registration, OrderedDict([
                (cls._prop_name, 'sample-dataset'),
                (cls._prop_description, 'This is a sample description for dataset'),
                (cls._prop_tags, {'sample-tag': 'tag-value'}),
                (cls._prop_create_new_version, True)
            ]))
        ])
        return template

    def __init__(self, workspace, definition, json_utility, error_utility):
        self._workspace = workspace
        self._definition = definition
        self._json_utility = json_utility
        self._error_utility = error_utility

    def create_and_register_dataset(self, validate):
        dataset = self._create_dataset(validate)
        dataset = self._add_properties(dataset, validate)
        dataset = self._register(dataset)
        return dataset

    def _create_dataset(self, validate):
        dataset_type = self._json_utility.try_get_value(
            self._definition, self._prop_dataset_type, None,
            lambda v: v in self._valid_dataset_types,
            'The value for "{}" must be one of {}.'.format(self._prop_dataset_type, self._valid_dataset_types))
        parameters = self._json_utility.try_get_value(
            self._definition, self._prop_parameters, {},
            lambda v: isinstance(v, dict),
            'Property "{}" must be specified.'.format(self._prop_parameters))

        if dataset_type == 'File':
            path = self._get_path(parameters)
            if is_dataprep_installed():
                return FileDatasetFactory.from_files(path, validate)
            return self._create_dataset_from_blocks([
                _Block.craft_get_file_block(path)
            ], FileDataset)

        if dataset_type == 'Tabular':
            return self._create_tabular(parameters, validate)

        raise RuntimeError('Unexpected code path for dataset_type: ' + dataset_type)

    def _add_properties(self, dataset, validate):
        time_series = self._json_utility.try_get_value(self._definition, self._prop_time_series, None)
        if time_series:
            fine_grain_timestamp = self._json_utility.try_get_value(
                time_series, self._prop_fine_grain_timestamp, None)
            coarse_grain_timestamp = self._json_utility.try_get_value(
                time_series, self._prop_coarse_grain_timestamp, None)
            if is_dataprep_installed():
                dataset = dataset.with_timestamp_columns(fine_grain_timestamp, coarse_grain_timestamp, validate)
            else:
                if fine_grain_timestamp:
                    dataset._properties[_DATASET_PROP_TIMESTAMP_FINE] = fine_grain_timestamp
                if coarse_grain_timestamp:
                    dataset._properties[_DATASET_PROP_TIMESTAMP_COARSE] = coarse_grain_timestamp
        return dataset

    def _register(self, dataset):
        registration = self._json_utility.try_get_value(
            self._definition, self._prop_registration, None,
            lambda v: isinstance(v, dict),
            'Property "{}" must be specified.'.format(self._prop_registration))
        name = self._json_utility.try_get_value(
            registration, self._prop_name, '',
            lambda v: len(v) > 0,
            'Property "{}.{}" must be specified.'.format(self._prop_registration, self._prop_name))
        description = self._json_utility.try_get_value(registration, self._prop_description, None)
        tags = self._json_utility.try_get_value(registration, self._prop_tags, None)
        create_new_version = self._json_utility.try_get_value(registration, self._prop_create_new_version, False)
        return dataset.register(self._workspace, name, description, tags, create_new_version)

    def _create_tabular(self, parameters, validate):
        source_type = self._json_utility.try_get_value(
            parameters, self._prop_source_type, None,
            lambda v: v in self._valid_source_types,
            'Property "{}" must be one of {}.'.format(self._prop_source_type, self._valid_source_types))

        if source_type == 'sql_query':
            query = self._get_query(parameters)
            if is_dataprep_installed():
                return TabularDatasetFactory.from_sql_query(query, validate)
            return self._create_dataset_from_blocks([
                _Block.craft_read_sql_block(query)
            ], TabularDataset)

        path = self._get_path(parameters)
        include_path = self._json_utility.try_get_value(
            parameters, self._prop_include_path, self._default_include_path)
        partition_format = self._json_utility.try_get_value(parameters, self._prop_partition_format, None)

        if source_type == 'parquet_files':
            if is_dataprep_installed():
                return TabularDatasetFactory.from_parquet_files(
                    path, validate, include_path, partition_format=partition_format)
            return self._create_dataset_from_blocks([
                _Block.craft_get_file_block(path),
                _Block.craft_read_parquet_block(),
                _Block.craft_partition_format_block(partition_format) if partition_format else None,
                _Block.craft_drop_path_column_block() if not include_path else None
            ], TabularDataset)

        if source_type == 'json_lines_files':
            if is_dataprep_installed():
                return TabularDatasetFactory.from_json_lines_files(
                    path, validate, include_path, partition_format=partition_format)
            return self._create_dataset_from_blocks([
                _Block.craft_get_file_block(path),
                _Block.craft_read_json_lines_block(),
                _Block.craft_partition_format_block(partition_format) if partition_format else None,
                _Block.craft_drop_path_column_block() if not include_path else None
            ], TabularDataset)

        if source_type == 'delimited_files':
            infer_column_types = self._json_utility.try_get_value(
                parameters, self._prop_infer_column_types, self._default_infer_column_types)
            separator = self._json_utility.try_get_value(parameters, self._prop_separator, self._default_separator)
            header = self._json_utility.try_get_value(parameters, self._prop_header, self._default_header)
            if is_dataprep_installed():
                return TabularDatasetFactory.from_delimited_files(
                    path, validate,
                    include_path=include_path,
                    partition_format=partition_format,
                    infer_column_types=infer_column_types,
                    separator=separator,
                    header=header)
            if infer_column_types:
                _raise_dataprep_missing_error(
                    'Cannot infer column types',
                    self._error_utility.get_error_message(
                        'setting {} to false'.format(self._prop_infer_column_types)))
            return self._create_dataset_from_blocks([
                _Block.craft_get_file_block(path),
                _Block.craft_read_delimited_block(separator, header),
                _Block.craft_partition_format_block(partition_format) if partition_format else None,
                _Block.craft_drop_path_column_block() if not include_path else None
            ], TabularDataset)

        raise RuntimeError('Unexpected code path for source_type: ' + source_type)

    def _get_path(self, props):
        path = self._json_utility.try_get_value(
            props, self._prop_path, [],
            lambda v: len([v] if not isinstance(v, list) else v) > 0,
            'Property "{}" must be specified.'.format(self._prop_path))
        if not isinstance(path, list):
            path = [path]
        return [self._resolve_path(p, is_query=False) for p in path]

    def _get_query(self, props):
        query = self._json_utility.try_get_value(
            props, self._prop_query, None,
            lambda v: v is not None,
            'Property "{}" must be specified.'.format(self._prop_query))
        return self._resolve_path(query, is_query=True)

    def _resolve_path(self, item, is_query=False):
        if not is_query and isinstance(item, str):
            return item

        item_prop = self._prop_query if is_query else self._prop_path
        subitem_prop = self._prop_query if is_query else self._prop_relative_path
        datastoreName = self._json_utility.try_get_value(
            item, self._prop_datastore_name, None,
            lambda v: isinstance(v, str) and len(v) > 0,
            'Property "{}.{}" must be specified.'.format(item_prop, self._prop_datastore_name)
        )
        subitem = self._json_utility.try_get_value(
            item, subitem_prop, None,
            lambda v: isinstance(v, str) and len(v) > 0,
            'Property "{}.{}" must be specified.'.format(item_prop, subitem_prop)
        )
        return (Datastore.get(self._workspace, datastoreName), subitem)

    def _create_dataset_from_blocks(self, block_list, cls):
        dflow_obj = {
            'blocks': [{
                'id': str(uuid4()),
                'type': b.block_type,
                'arguments': b.args
            } for b in block_list if b is not None]
        }
        return cls._create(definition=json.dumps(dflow_obj))


class _JsonUtility:
    def __init__(self, error_utility):
        self._error_utility = error_utility

    def try_get_value(self, props, key, default, validationFn=None, error=None):
        value = props.get(key, default)
        if validationFn and not validationFn(value):
            self._error_utility.raise_error(error or 'Invalid value for "{}": {}.'.format(key, value))
        return value


class _ErrorUtility:
    def __init__(self, def_path):
        self._def_path = def_path

    def raise_error(self, message):
        raise RuntimeError(self.get_error_message(message))

    def get_error_message(self, message):
        return message + ' (in definition file: "{}")'.format(self._def_path)


class _Block:
    def __init__(self, block_type, args):
        self.block_type = block_type
        self.args = args

    @staticmethod
    def craft_get_file_block(path):
        path = _validate_and_normalize_path(path)
        if isinstance(path[0], str):
            blob_pattern = re.compile(r'^https?://[^/]+\.blob\.core\.windows\.net', re.IGNORECASE)
            return _Block(
                'Microsoft.DPrep.GetFilesBlock', {
                    'path': {
                        'target': 1 if all(blob_pattern.match(p) for p in path) else 4,
                        'resourceDetails': [{'path': p} for p in path]
                    }
                })
        return _Block(
            'Microsoft.DPrep.GetDatastoreFilesBlock', {
                'datastores': [{
                    'subscription': p._datastore.workspace.subscription_id,
                    'resourceGroup': p._datastore.workspace.resource_group,
                    'workspaceName': p._datastore.workspace.name,
                    'datastoreName': p._datastore.name,
                    'path': p.path_on_datastore
                } for p in path]
            })

    @staticmethod
    def craft_read_sql_block(query):
        store, sql_query = query
        return _Block(
            'Microsoft.DPrep.ReadDatastoreSqlBlock', {
                'datastore': {
                    'subscription': store.workspace.subscription_id,
                    'resourceGroup': store.workspace.resource_group,
                    'workspaceName': store.workspace.name,
                    'datastoreName': store.name,
                    'path': sql_query
                }
            })

    @staticmethod
    def craft_read_parquet_block():
        return _Block('Microsoft.DPrep.ReadParquetFileBlock', {})

    @staticmethod
    def craft_read_json_lines_block():
        return _Block('Microsoft.DPrep.ParseJsonLinesBlock', {})

    @staticmethod
    def craft_read_delimited_block(separator, header):
        return _Block('Microsoft.DPrep.ParseDelimitedBlock', {
            'separator': separator,
            'columnHeadersMode': 3 if header is True else 0
        })

    @staticmethod
    def craft_partition_format_block(partition_format):
        return _Block('Microsoft.DPrep.AddColumnsFromPartitionFormatBlock', {
            'column': {
                'details': {'selectedColumn': 'Path'},
                'type': 2
            },
            'ignoreError': False,
            'partitionFormat': partition_format
        })

    @staticmethod
    def craft_drop_path_column_block():
        return _Block('Microsoft.DPrep.DropColumnsBlock', {
            'columns': {
                'details': {'selectedColumn': 'Path'},
                'type': 2
            }
        })


def _raise_dataprep_missing_error(issue, how_to_fix):
    error = get_dataprep_missing_message(issue, None, how_to_fix)
    raise RuntimeError(error)
