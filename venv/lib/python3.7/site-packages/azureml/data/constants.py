# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Constants used in the azureml.data package. Internal use only."""

AZURE_DATA_LAKE = "AzureDataLake"
AZURE_FILE = "AzureFile"
AZURE_BLOB = "AzureBlob"
AZURE_SQL_DATABASE = "AzureSqlDatabase"
AZURE_POSTGRESQL = "AzurePostgreSql"
AZURE_MYSQL = "AzureMySql"
DBFS = "DBFS"
AZURE_DATA_LAKE_GEN2 = "AzureDataLakeGen2"
HDFS = "Hdfs"
CUSTOM = "Custom"

ACCOUNT_KEY = "AccountKey"
SAS = "Sas"
CLIENT_CREDENTIALS = "ClientCredentials"
NONE = "None"
SQL_AUTHENTICATION = "SqlAuthentication"
SERVICE_PRINCIPAL = "ServicePrincipal"
MANAGED_DATASET = "managed-dataset"

STORAGE_RESOURCE_URI = "https://storage.azure.com/"
ADLS_RESOURCE_URI = "https://datalake.azure.net/"

WORKSPACE_BLOB_DATASTORE = "workspaceblobstore"
WORKSPACE_FILE_DATASTORE = "workspacefilestore"

ROLE_ID_OF_READER = 'acdd72a7-3385-48ef-bd42-f606fba81ae7'
ROLE_ID_OF_STORAGE_BLOB_DATA_READER = '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'

CONFLICT_MESSAGE = "Another datastore with the same name already exists"

_HYPERDRIVE_SUBMIT_ACTIVITY = "HyperDriveSubmit"
_SCRIPT_RUN_SUBMIT_ACTIVITY = "ScriptRunSubmit"
_PROFILE_RUN_SUBMIT_ACTIVITY = "DatasetProfileRunSubmit"

_SKIP_VALIDATE_DATASETS = "SkipValidateDatasets"

_AUTOML_SUBMIT_ACTIVITY = "AutoMLSubmit"
_AUTOML_INPUT_TYPE = "InputType"
_AUTOML_DATSET_ID = "DatasetId"
_AUTOML_COMPUTE = "Compute"
_AUTOML_DATASETS = "Datasets"
_AUTOML_SPARK = "IsSpark"
_AUTOML_DATASETS_COUNT = "DatasetCount"
_AUTOML_TABULAR_DATASETS_COUNT = "TabularDatasetCount"
_AUTOML_DATAFLOW_COUNT = "DataflowCount"
_AUTOML_PIPELINE_TABULAR_COUNT = "PipelineOutputTabularDatasetCount"
_AUTOML_OTHER_COUNT = "OtherCount"

_PUBLIC_API = 'PublicApi'

_DATASET_TYPE_TABULAR = 'tabular'
_DATASET_TYPE_FILE = 'file'

"""
Naming convention for trait property:
'_{trait}_{mark}{property}_'
    .format(
        trait='TimeSeries',
        mark='Column:' if True else ''
        property='FineGrainTimestamp')
"""
_DATASET_PROP_TIMESTAMP_FINE = '_TimeSeries_Column:FineGrainTimestamp_'
_DATASET_PROP_TIMESTAMP_COARSE = '_TimeSeries_Column:CoarseGrainTimestamp_'
_DEPRECATED_TIMESTAMP_NAME = 'fine_grain_timestamp'
_DEPRECATED_PARTITION_TIMESTAMP_NAME = 'coarse_grain_timestamp'
_DATASET_PROP_LABEL = '_Label_Column:Label_'
_DATASET_PROP_IMAGE = '_Image_Column:Image_'
_HALF_SECOND = 500000
_LABEL_COLUMN_NAME = 'column'
_IMAGE_URL_COLUMN_NAME = 'column'
_IMAGE_URL_COLUMN = 'Column'
_DATASET_PROP_DEPRECATED_BLOCKS = '_Dataflow_DeprecatedBlocks_'

_LOCAL_COMPUTE = 'local'
_ACTION_TYPE_PROFILE = 'profile'
_LEGACY_DATASET_ID = '00000000-0000-0000-0000-000000000000'
_PROFILE_RUN_ACTION_ID = '_azureml.DatasetActionId'

_TELEMETRY_ENTRY_POINT_DATASET = 'PythonSDK:Dataset'

_DATASET_ARGUMENT_TEMPLATE = 'DatasetConsumptionConfig:{}'
_DATASET_OUTPUT_ARGUMENT_TEMPLATE = 'DatasetOutputConfig:{}'
UPLOAD_MODE = 'upload'
DOWNLOAD_MODE = 'download'
MOUNT_MODE = 'mount'
DIRECT_MODE = 'direct'
HDFS_MODE = 'hdfs'
LINK_MODE = 'link'

_TIMESERIES_WITH_TIMESTAMP_COLUMN_ACTIVITY = "TimeSeriesDatasetAssignTimestampColumns"
_TIMESERIES_BEFORE_ACTIVITY = "TimeSeriesDatasetBefore"
_TIMESERIES_AFTER_ACTIVITY = "TimeSeriesDatasetAfter"
_TIMESERIES_BETWEEN_ACTIVITY = "TimeSeriesDatasetBetween"
_TIMESERIES_RECENT_ACTIVITY = "TimeSeriesDatasetRecent"

_PROHIBITED_NAMES = {
    'path',
    'ld_library_path',
    'user',
    'logname',
    'home',
    'pwd',
    'shell'
}
_PATITION_KEY_ACTIVITY = "PartitionKey"
_PATITION_KEY_VALUES_ACTIVITY = "PartitionKeyValues"
_PARTITION_KEYS = 'partition_keys'
_PATITION_BY_ACTIVITY = 'PartitionBy'
