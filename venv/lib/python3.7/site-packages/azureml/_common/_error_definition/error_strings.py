# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


class AzureMLErrorStrings:
    """
    All un-formatted error strings that accompany the common error codes in AzureML.
    """

    class UserErrorStrings:
        """
        Un-formatted error string for all UserErrors.

        Dev note: Please keep this list sorted on keys.
        """
        ACCOUNT_CONFIGURATION_CHANGED = "Configuration of your account was changed. {message}"
        ARGUMENT_BLANK_OR_EMPTY = "An empty value for argument [{argument_name}] is provided."
        ARGUMENT_INVALID = "Argument [{argument_name}] is of invalid type. Expected type: [{expected_type}]"
        ARGUMENT_INVALID_TYPE = "Value of type {type} is not supported, supported types include [{expected_type}]"
        ARGUMENT_MISMATCH = "Argument(s) [{argument_names}] has incompatible values: [{value_list}]."
        ARGUMENT_OUT_OF_RANGE = "Value for argument [{argument_name}] is out of range (Range: [{min} - {max}])."
        ARGUMENT_SIZE_OUT_OF_RANGE_TYPE = "Number size of type [{argument_name}] is out of range bits " \
                                          "(Range: [{min} - {max}])."
        AUTH = "Access to resource [{resource_name}] is prohibited. Please make sure the resource exists, " \
               "and/or you have the right permissions on it."
        AUTHENTICATION = "Authentication for [{resource_name}] failed. Please make sure the resource exists, " \
                         "and/or you have the right permissions on it."
        AUTHORIZATION = "Authorization for [{resource_name}] failed. Please make sure the resource exists, " \
                        "and/or you have the right permissions on it."
        AUTHORIZATION_BLOB_STORAGE = "Encountered authorization error while uploading to blob storage. Please " \
                                     "check the storage account attached to your workspace. Make sure " \
                                     "that the current user is authorized to access the storage account " \
                                     "and that the request is not blocked by a firewall, virtual network, " \
                                     "or other security setting.\n" \
                                     "\tStorageAccount: {account_name}\n\tContainerName: {container_name}" \
                                     "\n\tStatusCode: {status_code}"  # "\n\tErrorCode: {error_code}"
        BAD_ARGUMENT = "An invalid value for argument [{argument_name}] was provided."
        BAD_DATA = "[{data_argument_name}] was invalid."
        BAD_DATA_DOWNLOAD = "Downloaded file did not match blob: {file_size} bytes downloaded but " \
                            "{content_length} bytes present in blob."
        BAD_DATA_UPLOAD = "Uploaded file did not match local file: {content_length} bytes " \
                          "uploaded but {file_size} bytes present locally."
        BLOB_NOT_FOUND = "Download of file failed with error. The specified blob does not exist."
        COMPUTE_NOT_FOUND = "Compute [{compute_name}] was not found in workspace [{workspace_name}]."
        CONFLICT = "[{resource_name}] is in a conflicting state."
        CONNECTION_FAILURE = "Connection to [{resource_name}] failed."
        CREATE_CHILDREN_FAILED = "Failed to create children {run_id}"
        CREDENTIAL_EXPIRED_INACTIVITY = "Credentials have expired due to inactivity. {message}"
        CREDENTIAL_EXPIRED_PASSWORD_CHANGE = "The credential data used by CLI has been expired because you might have"\
                                             " changed or reset the password. {message}"
        DEPRECATED = "[{feature_name}] is deprecated."
        DISABLED = "[{resource_name}] is disabled."
        DOWNLOAD_FAILED = "Download of file failed with error: {error}"
        DUPLICATE_ARGUMENT = "Argument [{argument_name}] has duplicate values: [{values}]."
        EMPTY_DATA = "[{data_argument_name}] was invalid."
        EXPERIMENT_NOT_FOUND = "Experiment [{experiment_name}] was not found in workspace [{workspace_name}]."
        FILE_ALREADY_EXISTS = "Specified output file {full_path} already exists"
        FAILED_ID_WITHIN_SECONDS = "Delete experiment with id {experiment_id} failed to return success " \
                                   "within {timeout_seconds} seconds"
        IMMUTABLE = "[{resource_name}] cannot be modified because it's immutable."
        INVALID_COLUMN_DATA = "Rows cannot contain {type} values, column {column} contains a {type}."
        INVALID_COLUMN_LENGTH = "Columns must have the same length, column {reference_column} had " \
                                "length {table_column_length}, however, column {key} had length {column_length}."
        INVALID_DATA = "[{data_argument_name}] is invalid."
        INVALID_DIMENSION = "[{data_argument_name}] dimension does not have required dimensions."
        INVALID_OUTPUT_STREAM = "Trying to access output stream redirector but there is none. " \
                                "If you want to redirect output streams, set redirect_output_stream to True"
        INVALID_STATUS = "Run does not have process information yet."
        INVALID_URI = "This run does not have a diagnostics Uri to download diagnostics."
        KEY_VAULT_NOT_FOUND = "KeyVault [{keyvault_name}] is not found in workspace [{workspace_name}]."
        MALFORMED_ARGUMENT = "Argument [{argument_name}] is malformed."
        MEMORY = "Insufficient memory to execute the request. Please retry on a virtual machine with more memory."
        METRICS_NUMBER_EXCEEDS = "Number of metrics {metric_dtos} is greater than " \
                                 "the max number of metrics that should be " \
                                 "sent in batch {AZUREML_MAX_NUMBER_METRICS_BATCH}"
        MISSING_DATA = "[{data_argument_name}] has missing data."
        METHOD_ALREADY_REGISTERED = "A submit function has already been registered on {method_class} class."
        METHOD_NOT_REGISTERED = "Method to be submitted has not been registered."
        NETWORK_CONNECTION_FAILURE = "Please ensure you have network connection. Error details: {error}"
        NOT_FOUND = "Resource [{resource_name}] was not found."
        NOT_READY = "[{resource_name}] is not ready."
        NOT_SUPPORTED = "Request for [{scenario_name}] is not supported."
        NOT_SUPPORTED_FOR_SCENARIO = "Setting [{param}] is not supported for [{scenario_name}]."
        ONLY_SUPPORTED_SERVICE_SIDE_FILTERING = "Recursive get_child_runs is only supported with " \
                                                "service side filtering for root runs"
        QUOTA_EXCEEDED = "Quota exceeded for [{resource_name}]."
        RESOURCE_EXHAUSTED = "Insufficient resource for [{resource_name}] to execute the request."
        START_CHILDREN_FAILED = "Failed to start children {run_id}"
        STORAGE_ACCOUNT_NOT_FOUND = "Storage account [{storage_account_name}] was not found in workspace " \
                                    "[{workspace_name}]."
        SSL_ERROR = "Certificate verification failed. This typically happens when using Azure CLI behind a proxy " \
                    "that intercepts traffic with a self-signed certificate. " \
                    "Please add this certificate to the trusted CA bundle. " \
                    "More info: https://docs.microsoft.com/cli/azure/use-cli-effectively#work-behind-a-proxy."
        TIMEOUT = "Operation timed out."
        TIMEOUT_FLUSH_TASKS = "Failed to flush task queue within {timeout_seconds} seconds"
        TOO_MANY_REQUESTS = "Received too many requests in a short amount of time. Please retry again later."
        TWO_INVALID_ARGUMENT = "Invalid parameters, one of {arg_one} and {arg_two} is required as input"
        TWO_INVALID_PARAMETER = "Invalid parameters, {arg_one} and {arg_two} were both provided, " \
                                "only one at a time is supported"
        UNSUPPORTED_RETURN_TYPE = "Unsupported return type {return_object} from command"
        WORKSPACE_NOT_FOUND = "Workspace [{workspace_name}] was not found."

    class SystemErrorStrings:
        """
        Un-formatted error string for all SystemErrors.

        Dev note: Please keep this list sorted on keys.
        """
        CLIENT_ERROR = "Failed to process request. If you think this is a bug, " \
                       "please create a support request quoting client unique identifier [{client_request_id}]"
        SERVICE_ERROR = "Failed to process request. If you think this is a bug, " \
                        "please create a support request quoting service unique identifier [{server_request_id}]"
