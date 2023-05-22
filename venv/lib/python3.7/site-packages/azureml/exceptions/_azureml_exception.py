# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._common.exceptions import AzureMLException
from azureml._common._error_response._error_response_constants import ErrorCodes


class TrainingException(AzureMLException):
    """An exception related to failures in configuring, running, or
    updating a training run. Validation of the run configuration covers most of
    the uses of the exception. Other possible sources can be incorrect
    metric names in a training script, which can cause downstream errors during
    hyper parameter tuning.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of TrainingException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(TrainingException, self).__init__(exception_message, **kwargs)


class ExperimentExecutionException(AzureMLException):
    """
    An exception related to failures in configuring, running, or
    updating a submitted run. Validation of the run configuration covers most of
    the uses of this exception. Other possible sources can be failures when
    submitting the experiment run.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of ExperimentExecutionException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(ExperimentExecutionException, self).__init__(exception_message, **kwargs)


class ProjectSystemException(AzureMLException):
    """
    An exception related to failures while downloading snapshotted project
    files, reading from, or setting up the source directory. This exception is commonly raised
    to indicate invalid or missing scope information for an experiment and workspace.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of ProjectSystemException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(ProjectSystemException, self).__init__(exception_message, **kwargs)


class SnapshotException(AzureMLException):
    """
    An exception related to failures while snapshotting a project. This exception is commonly raised
    when taking a snapshot when the run already has one, and for directory size and file count limit issues.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of SnapshotException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(SnapshotException, self).__init__(exception_message, **kwargs)


class RunEnvironmentException(AzureMLException):
    """
    An exception related to missing or invalid information related to loading a
    Run from the current environment. This exception is commonly raised when attempting to
    load a run outside of an execution context.
    """

    def __init__(self, **kwargs):
        super(RunEnvironmentException, self).__init__(
            ("Could not load a submitted run, if outside "
             "of an execution context, use experiment.start_logging to "
             "initialize an azureml.core.Run."), **kwargs)


class WorkspaceException(AzureMLException):
    """
    An exception related to failures creating or getting a workspace.
    This exception is commonly raised related to permissions and resourcing.
    Ensure that the resource group exists if specified and that the
    authenticated user has access to the subscription.

    Workspace names are unique within a resource group.

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param found_multiple: Specifies whether multiple workspaces were found.
    :type found_multiple: bool
    """

    def __init__(self, exception_message, found_multiple=False, **kwargs):
        """
        Initialize a new instance of WorkspaceException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param found_multiple: Specifies whether multiple workspaces were found.
        :type found_multiple: bool
        """
        super(WorkspaceException, self).__init__(exception_message, **kwargs)
        self.found_multiple = found_multiple


class WorkspacePrivateEndpointException(AzureMLException):
    """
    An exception related to failures creating or getting a workspace private endpoint.

    Workspace names are unique within a resource group.

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param found_multiple: Specifies whether multiple workspaces were found.
    :type found_multiple: bool
    """

    def __init__(self, exception_message, found_multiple=False, **kwargs):
        """
        Initialize a new instance of WorkspacePrivateEndpointException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param found_multiple: Specifies whether multiple workspaces were found.
        :type found_multiple: bool
        """
        super(WorkspacePrivateEndpointException, self).__init__(exception_message, **kwargs)
        self.found_multiple = found_multiple


class ComputeTargetException(AzureMLException):
    """
    An exception related to failures when creating, interacting with, or
    configuring a compute target. This exception is commonly raised for
    failures attaching a compute target, missing headers, and unsupported
    configuration values.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of ComputeTargetException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(ComputeTargetException, self).__init__(exception_message, **kwargs)


class UserErrorException(AzureMLException):
    """
    An exception related to invalid or unsupported inputs. This exception is commonly raised for
    missing parameters, trying to access an entity that does not exist, or invalid value types
    when configuring a run.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    _error_code = ErrorCodes.USER_ERROR

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of UserErrorException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(UserErrorException, self).__init__(exception_message, **kwargs)


class AuthenticationException(UserErrorException):
    """
    An exception related to failures in authenticating. The general solution to
    most instances of this exception is to try ``az login`` to authenticate
    through the browser. Other sources for the exception include invalid or
    unspecified subscription information. Trying ``az account set -s {subscription_id}``
    to specify the subscription usually resolves missing or ambiguous
    subscription errors. For more information about using the ``az`` command, see
    https://docs.microsoft.com/cli/azure/authenticate-azure-cli.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    _error_code = ErrorCodes.AUTHENTICATION_ERROR

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of AuthenticationException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(AuthenticationException, self).__init__(exception_message, **kwargs)


class RunConfigurationException(AzureMLException):
    """
    An exception related to failures in locating or serializing a run
    configuration. This exception is commonly raised when passing unsupported
    values to the run configuration. Failures in deserialization can be exposed
    after the fact while trying to submit even if the problem was introduced earlier.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    def __init__(self, exception_message, **kwargs):
        """
        Initialize a new instance of RunConfigurationException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        """
        super(RunConfigurationException, self).__init__(exception_message, **kwargs)


class WebserviceException(AzureMLException):
    """
    An exception related to failures while interacting with model management
    service. This exception is commonly raised for failed REST requests to
    model management service.

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param status_code: An optional HTTP status code that describes the web service request.
    :type status_code: str
    """

    def __init__(self, exception_message, status_code=None, logger=None, **kwargs):
        """
        Initialize a new instance of WebserviceException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param status_code: An optional HTTP status code that describes the web service request.
        :type status_code: str
        """
        super(WebserviceException, self).__init__(exception_message, **kwargs)
        self.status_code = status_code

        if logger:
            try:
                logger.error(exception_message + '\n')
            except Exception:
                pass


class ModelNotFoundException(AzureMLException):
    """
    An exception related to missing model when attempting to download a
    previously registered model. This exception is commonly raised when
    trying to download a model from a different workspace, with the wrong name,
    or an invalid version.

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param logger: An optional logger to which to send the exception message.
    :type logger: logging.logger
    """

    def __init__(self, exception_message, logger=None, **kwargs):
        """
        Initialize a new instance of ModelNotFoundException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param logger: An optional logger to which to send the exception message.
        :type logger: logger
        """
        super(ModelNotFoundException, self).__init__(exception_message, **kwargs)

        if logger:
            try:
                logger.error(exception_message + '\n')
            except Exception:
                pass


class ModelPathNotFoundException(AzureMLException):
    """
    An exception related to missing model files when registering a model.
    This exception is commonly raised when trying to register a model before
    uploading the model files. Calling a run's :meth:`azureml.core.Run.upload_file` method
    before attempting to register the model usually resolves the error.

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param logger: An optional logger to which to send the exception message.
    :type logger: logger
    """

    def __init__(self, exception_message, logger=None, **kwargs):
        """
        Initialize a new instance of ModelPathNotFoundException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param logger: An optional logger to which to send the exception message.
        :type logger: logger
        """
        super(ModelPathNotFoundException, self).__init__(exception_message, **kwargs)

        if logger:
            try:
                logger.error(exception_message)
            except Exception:
                pass


class DiscoveryUrlNotFoundException(AzureMLException):
    """
    An exception related to an unsuccessful loading of the location from the
    environment. If the environment variable is not found, the SDK makes a
    request to ARM to retrieve the URL.

    :param discovery_key: The name of the URL not found.
    :type discovery_key: str
    """

    def __init__(self, discovery_key, **kwargs):
        """
        Initialize a new instance of DiscoveryUrlNotFoundException.
        :param discovery_key: The name of the URL not found.
        :type discovery_key: str
        """
        super(DiscoveryUrlNotFoundException, self).__init__(
            "Could not load discovery key {}, from environment variables.".format(discovery_key), **kwargs)


class ActivityFailedException(AzureMLException):
    """
    An exception related to failures in an activity. This exception is commonly raised
    for failures during submitted experiment runs. The failure
    is generally seen from the control pane specifically during a
    wait_for_completion call on an activity, however the source for the failure
    is usually within the activity's logic.

    :param error_details: A description of the error.
    :type error_details: str
    """

    def __init__(self, error_details, **kwargs):
        """
        Initialize a new instance of ActivityFailedException.

        :param error_details: A description of the error.
        :type error_details: str
        """
        super(ActivityFailedException, self).__init__("Activity Failed:\n{}".format(error_details), **kwargs)


class ActivityCanceledException(AzureMLException):
    """
    An exception capturing a run that was canceled. The exception
    is generally seen from the control plane specifically during a
    wait_for_completion call on an activity.

    :param cancellation_source: An optional message describing the process canceled the activity.
    :type cancellation_source: str
    """

    def __init__(self, cancellation_source=None, **kwargs):
        """
        Initialize a new instance of ActivityCanceledException.
        :param cancellation_source: An optional message describing the process canceled the activity.
        :type cancellation_source: str
        """
        super(ActivityCanceledException, self).__init__("Activity Canceled", **kwargs)


class DatasetTimestampMissingError(AzureMLException):
    """
    An exception indicating that an expected timestamp column is not assigned.
    The exception is raised in the Tabular Dataset instance when time series related APIs are called
    but the expected timestamp columns cannot be found, for example, because they were not set or
    they were dropped.

    :param error_details: A description of the error.
    :type error_details: str
    """
    def __init__(self, error_details, **kwargs):
        """
        Initialize a new instance of ActivityFailedException.

        :param error_details: A description of the error.
        :type error_details: str
        """
        super(DatasetTimestampMissingError, self).__init__("Activity Failed:\n{}".format(error_details), **kwargs)


class ParallelRunException(AzureMLException):
    """
    An exception used in ParallelRunStep.
    Users can customize the retry logic by raising this exception with the code to indict the intention.

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param code: Specifies the intended retry logic. Can be one of "Retry.NoRetry",
        "Retry.StopProcess" or "Retry.RestartProcess".
    :type code: str
    :param delay_seconds: Seconds to sleep before processing the next mini batch.
    :type delay_seconds: float
    """

    def __init__(self, exception_message: str, code: str = None, delay_seconds: float = 0, **kwargs):
        """
        Initialize a new instance of ParallelRunException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param code: Specifies the intended retry logic. Can be one of "Retry.NoRetry",
            "Retry.StopProcess" or "Retry.RestartProcess" or "Retry.Sleep".
            If no code is provided, it would be considered equivalent to "Retry.Sleep"
            The handling of the codes are:
            Retry.NoRetry: The mini-batch would not be retried, in constrast with the normal behavior
                of trying to process the mini-batch for `run_max_try` times until succeeds as specified
                in ParallelRunConfig.
                This code is normally used when you think the failure can't be recovered by retrying.
            Retry.RestartProcess: The parallel run worker process would be restarted.
                This code is normally used when you think the process needs restarting to recover the failure.
            Retry.StopProcess: The parallel run worker process would be stopped.
                This code is normally used when you think there're too many worker processes that the host machine
                is stressed on resources and thus need to lower the workload to recover.
            Retry.Sleep: The worker process would sleep `delay_seconds` seconds before processing the next mini-batch.
                This code is normally used when you think the process should wait some time until its
                dependency recovers, such like external web service.
        :type code: str
        :param delay_seconds: Seconds to sleep before processing the next mini batch.
        :type delay_seconds: float
        """
        if code:
            assert code in ["Retry.NoRetry", "Retry.StopProcess", "Retry.RestartProcess", "Retry.Sleep"]

        super(ParallelRunException, self).__init__(exception_message, **kwargs)
        self.code = code
        self.delay_seconds = delay_seconds
