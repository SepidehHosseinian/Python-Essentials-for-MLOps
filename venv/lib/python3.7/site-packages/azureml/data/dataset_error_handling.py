# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains exceptions for dataset error handling in Azure Machine Learning."""
from builtins import getattr
from azureml._common.exceptions import AzureMLException
from azureml.data._dataprep_helper import dataprep, ensure_dataflow
from azureml.exceptions import UserErrorException
from azureml.data._loggerfactory import trace_error


class DatasetValidationError(UserErrorException):
    """Defines an exception for Dataset validation errors.

    :param message: The error message.
    :type message: str
    :param exception: The exception that caused this error.
    :type exception: Exception
    """

    def __init__(self, message, exception=None):
        """Class DatasetValidationError constructor.

        :param message: The error message.
        :type message: str
        :param exception: The exception that caused this error.
        :type exception: Exception
        """
        self.inner_exception = exception
        if exception is not None:
            self.error_code = getattr(exception, 'error_code', None)
            self.validation_error_code = getattr(exception, 'validation_error_code', None)
            self.compliant_message = getattr(exception, 'compliant_message', None)
        super().__init__(message)


class DatasetExecutionError(UserErrorException):
    """Defines an exception for Dataset execution errors.

    :param message: The error message.
    :type message: str
    :param exception: The exception that caused this error.
    :type exception: Exception
    """

    def __init__(self, message, exception):
        """Class DatasetExecutionError constructor.

        :param message: The error message.
        :type message: str
        :param exception: The exception that caused this error.
        :type exception: Exception
        """
        self.inner_exception = exception
        self.error_code = getattr(exception, 'error_code', None)
        self.validation_error_code = getattr(exception, 'validation_error_code', None)
        self.compliant_message = getattr(exception, 'compliant_message', None)

        super().__init__(message)


def _validate_has_data(dataflow, error_message):
    ensure_dataflow(dataflow)
    try:
        dataflow.verify_has_data()
    except (dataprep().api.dataflow.DataflowValidationError,
            dataprep().api.errorhandlers.ExecutionError) as e:
        raise DatasetValidationError(error_message + '\n' + e.compliant_message, exception=e)


def _validate_has_columns(dataflow, columns, expected_types=None):
    if expected_types is not None and len(columns) != len(expected_types):
        raise UserErrorException('Length of `columns` and `expected_types` must be the same')
    ensure_dataflow(dataflow)
    profile = dataflow.keep_columns(columns).take(1)._get_profile()
    if profile.row_count == 0 or profile.row_count is None:
        missing_columns = columns
    else:
        missing_columns = [col for col in columns if col not in profile.columns.keys()]
    if missing_columns:
        raise DatasetValidationError('The specified columns {} do not exist in the current dataset.'
                                     .format(missing_columns))
    if not expected_types:
        return
    mismatch_columns = []
    mismatch_types = []
    for i in range(len(columns)):
        if profile.columns[columns[i]].type != expected_types[i]:
            mismatch_columns.append(columns[i])
            mismatch_types.append(str(expected_types[i])[10:])
    if mismatch_columns:
        raise DatasetValidationError('The specified columns {} do not have the expected types {}.'
                                     .format(mismatch_columns, mismatch_types))


def _handle_dataset_exception(e, dataset_info=None, operation=None):
    message, is_dprep_exception = _construct_message_and_check_exception_type(e, dataset_info, operation)
    _dataprep_error_handler(e, message, is_dprep_exception)


def _try_execute(action, operation=None, dataset_info=None, **kwargs):
    try:
        if len(kwargs) > 0:
            return action(kwargs)
        else:
            return action()
    except Exception as e:
        _handle_dataset_exception(e, dataset_info, operation)


def _construct_message_and_check_exception_type(e, dataset_info, operation):
    from azureml.dataprep import DataPrepException
    is_dprep_exception = False
    if dataset_info is not None:
        id = dataset_info.get('id', '')
        name = dataset_info.get('name', '')
        version = dataset_info.get('version', '')

        if isinstance(e, DataPrepException):
            message = ("Execution failed in operation '{}' for Dataset(id='{}', name='{}', version={}, error_code={},"
                       "error_message={})".format(
                           operation,
                           id,
                           name,
                           version,
                           getattr(e, 'error_code', None),
                           getattr(e, 'compliant_message', None)))
            is_dprep_exception = True
        else:
            message = ("Execution failed in operation '{}' for Dataset(id='{}', name='{}', version={}, error_code={}, "
                       "exception_type={})".format(
                           operation,
                           id,
                           name,
                           version,
                           getattr(e, 'error_code', None),
                           e.__class__.__name__))
    else:
        if isinstance(e, DataPrepException):
            message = "Execution failed with error message: {}".format(e.compliant_message)
            is_dprep_exception = True
        else:
            message = "Execution failed unexpectedly due to: {}".format(e.__class__.__name__)

    return message, is_dprep_exception


def _dataprep_error_handler(e, message, is_dprep_exception):
    user_exception_list = ["Authentication", "NotFound", "Validation", "FieldNotFound",
                           "AlreadyExists", "FieldConflict", "StepTranslation", "DataError",
                           "NoColumnsError", "Assertion", "AuthenticationContextMismatch",
                           "CreateTable", "WriteTable", "Py4JError"]

    if is_dprep_exception:
        message = message + " ErrorCode: {}".format(e.error_code)
        for item in user_exception_list:
            if _contains(item, getattr(e, 'error_code', 'Unexpected')):
                raise UserErrorException(message, inner_exception=e)

    raise AzureMLException(message, inner_exception=e)


def _contains(needle, haystack):
    if needle.lower() in haystack.lower():
        return True
    return False


def _download_error_handler(error_list, logger):
    def is_user_error(error: str):
        return error in [
            "Microsoft.DPrep.ErrorValues.SourceFileNotFound",
            "Microsoft.DPrep.ErrorValues.SourceFilePermissionDenied",
            "Microsoft.DPrep.ErrorValues.InvalidArgument",
            "Microsoft.DPrep.ErrorValues.ValueWrongKind",
            "Microsoft.DPrep.ErrorValues.SourcePermissionDenied",
            "Microsoft.DPrep.ErrorValues.DestinationPermissionDenied",
            "Microsoft.DPrep.ErrorValues.DestinationDiskFull",
            "Microsoft.DPrep.ErrorValues.FileSizeChangedWhileDownloading",
            "Microsoft.DPrep.ErrorValues.StreamInfoInvalidPath",
            "Microsoft.DPrep.ErrorValues.NoManagedIdentity",
            "Microsoft.DPrep.ErrorValues.NoOboEndpoint",
            "Microsoft.DPrep.ErrorValues.StreamInfoRequired"
        ]

    message = 'Some files have failed to download:' + '\n'.join(
        [str((file_name, error_code)) for (file_name, error_code) in error_list])
    all_user_errors = True
    for (_, error) in error_list:
        if not is_user_error(error):
            all_user_errors = False
            trace_error(logger, "System error happens during downloading: {}".format(error))
            break
    raise UserErrorException(message) if all_user_errors else AzureMLException(message)
