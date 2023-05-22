# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

MMS_WORKSPACE_API_VERSION = '2018-11-19'
MMS_ROUTE_API_VERSION = 'v1.0'
MMS_MIGRATION_TIMEOUT_SECONDS = 30
MMS_SYNC_TIMEOUT_SECONDS = 80
MMS_SERVICE_VALIDATE_OPERATION_TIMEOUT_SECONDS = 30
MMS_SERVICE_EXIST_CHECK_SYNC_TIMEOUT_SECONDS = 5
MMS_RESOURCE_CHECK_SYNC_TIMEOUT_SECONDS = 15
SUPPORTED_RUNTIMES = {
    'spark-py': 'SparkPython',
    'python': 'Python',
    'python-slim': 'PythonSlim'
}
UNDOCUMENTED_RUNTIMES = ['python-slim']
CUSTOM_BASE_IMAGE_SUPPORTED_RUNTIMES = {
    'python': 'PythonCustom'
}
WORKSPACE_RP_API_VERSION = '2019-06-01'
MAX_HEALTH_CHECK_TRIES = 30
HEALTH_CHECK_INTERVAL_SECONDS = 1
DOCKER_IMAGE_TYPE = "Docker"
UNKNOWN_IMAGE_TYPE = "Unknown"
WEBAPI_IMAGE_FLAVOR = "WebApiContainer"
ACCEL_IMAGE_FLAVOR = "AccelContainer"
IOT_IMAGE_FLAVOR = "IoTContainer"
UNKNOWN_IMAGE_FLAVOR = "Unknown"
CLOUD_DEPLOYABLE_IMAGE_FLAVORS = [WEBAPI_IMAGE_FLAVOR, ACCEL_IMAGE_FLAVOR]
SUPPORTED_CUDA_VERSIONS = ["9.0", "9.1", "10.0"]
ARCHITECTURE_AMD64 = "amd64"
ARCHITECTURE_ARM32V7 = "arm32v7"
ACI_WEBSERVICE_TYPE = "ACI"
AKS_WEBSERVICE_TYPE = "AKS"
AKS_ENDPOINT_TYPE = "AKSENDPOINT"
SERVICE_REQUEST_OPERATION_CREATE = "Create"
SERVICE_REQUEST_OPERATION_UPDATE = "Update"
SERVICE_REQUEST_OPERATION_DELETE = "Delete"
AKS_ENDPOINT_DELETE_VERSION = "deleteversion"
AKS_ENDPOINT_CREATE_VERSION = "createversion"
AKS_ENDPOINT_UPDATE_VERSION = "updateversion"
COMPUTE_TYPE_KEY = "computeType"
LOCAL_WEBSERVICE_TYPE = "Local"
UNKNOWN_WEBSERVICE_TYPE = "Unknown"
CLI_METADATA_FILE_WORKSPACE_KEY = 'workspaceName'
CLI_METADATA_FILE_RG_KEY = 'resourceGroupName'
MODEL_METADATA_FILE_ID_KEY = 'modelId'
IMAGE_METADATA_FILE_ID_KEY = 'imageId'
DOCKER_IMAGE_HTTP_PORT = 5001
DOCKER_IMAGE_MQTT_PORT = 8883
RUN_METADATA_EXPERIMENT_NAME_KEY = '_experiment_name'
RUN_METADATA_RUN_ID_KEY = 'run_id'
PROFILE_METADATA_CPU_KEY = "cpu"
PROFILE_METADATA_MEMORY_KEY = "memoryInGB"
PROFILE_PARTIAL_SUCCESS_MESSAGE = 'Model Profiling operation completed, but encountered issues ' +\
                                  'during execution: %s Inspect ModelProfile.error property for more information.'
PROFILE_FAILURE_MESSAGE = 'Model Profiling operation failed with the following error: %s Request ID: %s. ' +\
                          'Inspect ModelProfile.error property for more information.'
DATASET_SNAPSHOT_ID_FORMAT = '/datasetId/{dataset_id}/datasetSnapshotName/{dataset_snapshot_name}'
DOCKER_IMAGE_APP_ROOT_DIR = '/var/azureml-app'
IOT_WEBSERVICE_TYPE = "IOT"
MIR_WEBSERVICE_TYPE = "MIR"
MODEL_PACKAGE_ASSETS_DIR = 'azureml-app'
MODEL_PACKAGE_MODELS_DIR = 'azureml-models'
# Kubernetes namespaces must be valid lowercase DNS labels.
# Ref: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/architecture/identifiers.md
NAMESPACE_REGEX = '^[a-z0-9]([a-z0-9-]{,61}[a-z0-9])?$'
WEBSERVICE_SCORE_PATH = '/score'
WEBSERVICE_SWAGGER_PATH = '/swagger.json'
ALL_WEBSERVICE_TYPES = [ACI_WEBSERVICE_TYPE, AKS_ENDPOINT_TYPE, AKS_WEBSERVICE_TYPE, IOT_WEBSERVICE_TYPE,
                        MIR_WEBSERVICE_TYPE]
HOW_TO_USE_ENVIRONMENTS_DOC_URL = "https://docs.microsoft.com/azure/machine-learning/how-to-use-environments"

# Blob storage allows 10 minutes/megabyte. Add a bit for overhead, and to make sure we don't give up first.
# Ref: https://docs.microsoft.com/en-us/rest/api/storageservices/setting-timeouts-for-blob-service-operations
ASSET_ARTIFACTS_UPLOAD_TIMEOUT_SECONDS_PER_BYTE = 1.1 * (10 * 60) / (1000 * 1000)
