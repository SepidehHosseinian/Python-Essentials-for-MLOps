# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import sys
import traceback
import warnings

import azureml._common._error_response.error_hierarchy as error_hierarchy
from azureml._common._error_definition import AzureMLError


class AzureMLException(Exception):
    """
    The base class for all Azure Machine Learning exceptions.

    This class extends the Python Exception class. If you are trying to catch only Azure ML exceptions,
    then catch them with this class.

    .. remarks::

        The following code example shows how to handle the :class:`azureml.exceptions.WebserviceException`,
        which is a subclass of AzureMLException. In the code, if the check of the service fails, then
        the WebserviceException is handled and a message can be printed.

        .. code-block:: python

            from azureml.core.model import InferenceConfig
            from azureml.core.webservice import AciWebservice


            service_name = 'my-custom-env-service'

            inference_config = InferenceConfig(entry_script='score.py', environment=environment)
            aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

            service = Model.deploy(workspace=ws,
                                  name=service_name,
                                  models=[model],
                                  inference_config=inference_config,
                                  deployment_config=aci_config,
                                  overwrite=True)
            service.wait_for_deployment(show_output=True)

    :param exception_message: A message describing the error.
    :type exception_message: str
    :param inner_exception: An optional error message, for example, from a previously handled exception.
    :type inner_exception: str
    :param target: The name of the element that caused the exception to be thrown.
    :type target: str
    :param details: Any additional information for the error, such as other error responses or stacktraces.
    :type details: builtin.list(str)
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 target=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        Exception.__init__(self, exception_message)
        self._exception_message = exception_message
        self._inner_exception = inner_exception
        self._exc_info = sys.exc_info()
        self._target = target
        self._details = details
        self._message_format = message_format
        self._message_parameters = message_parameters
        self._reference_code = reference_code
        self._azureml_error = kwargs.pop("azureml_error", None)

        AzureMLException._warn_deprecations(message_format, "message_format")
        AzureMLException._warn_deprecations(message_parameters, "message_parameters")
        AzureMLException._warn_deprecations(reference_code, "reference_code")

        """
        Initialize a new instance of AzureMLException.

        :param exception_message: A message describing the error.
        :type exception_message: str
        :param inner_exception: An optional error message, for example, from a previously handled exception.
        :type inner_exception: str
        :param target: The name of the element that caused the exception to be thrown.
        :type target: str
        :param details: Any additional information for the error, such as other error responses or stacktraces.
        :type details: builtin.list(str)
        :param message_format: Un-formatted version of the exception_message with no variable substitution.
        :type message_format: str
        :param message_parameters: Value substitutions corresponding to the contents of message_format
        :type message_parameters: Dictionary[str, str]
        :param reference_code: Indicator of the module or code where the failure occurred
        :type reference_code: str
        """

    def __repr__(self):
        """Return the string representation of the AzureMLException object.

        :return: String representation of the AzureMLException object
        :rtype: str
        """
        return "{}:\n\tMessage: {}\n\tInnerException {}\n\tErrorResponse \n{}".format(
            self.__class__.__name__,
            self.message,
            self.inner_exception,
            self._serialize_json(indent=4)
        )

    def __str__(self):
        """Return the string representation of the AzureMLException."""
        return "{}:\n\tMessage: {}\n\tInnerException {}\n\tErrorResponse \n{}".format(
            self.__class__.__name__,
            self.message,
            self.inner_exception,
            self._serialize_json(
                indent=4, filter_fields=[AzureMLError.Keys.MESSAGE_FORMAT,
                                         AzureMLError.Keys.MESSAGE_PARAMETERS]
            )
        )

    @property
    def message(self):
        """Return the error message.

        :return: The error message.
        :rtype: str
        """
        return self._exception_message

    @message.setter
    def message(self, value):
        self._exception_message = value

    @property
    def inner_exception(self):
        """Return the inner exception message.

        :return: The inner exception message.
        :rtype: str
        """
        return self._inner_exception

    @inner_exception.setter
    def inner_exception(self, value):
        self._inner_exception = value

    def print_stacktrace(self):
        traceback.print_exception(*self._exc_info)

    @classmethod
    def _with_error(cls, azureml_error, **kwargs) -> "AzureMLException":
        """
        Method to create an instance of this class, from an instance of
        :class:`azureml._common._error_definition.azureml_error.AzureMLError`
        """
        exception = cls(exception_message=azureml_error.error_message, azureml_error=azureml_error, **kwargs)
        return exception

    def _serialize_json(self, indent=None, filter_fields=None):
        """
        Serialize this exception as an ErrorResponse JSON.
        :return: A JSON string representation of the exception.
        :rtype: str
        """
        if filter_fields is None:
            filter_fields = []

        error_ret = {}

        if self._azureml_error:
            root_error_dict = self._azureml_error.to_root_error_dict()
            root_error_dict = {k: v for (k, v) in root_error_dict.items() if v and k not in filter_fields}
            error_ret["error"] = root_error_dict
            return json.dumps(error_ret, indent=indent)

        error_out = {}
        for super_class in self.__class__.mro():
            if super_class.__name__ == AzureMLException.__name__:
                break
            try:
                error_code = getattr(super_class, '_error_code')
                if error_out != {}:
                    # Flatten the tree in case we have something like System > System > System > System
                    prev_code = error_out.get('code')
                    if prev_code is None or error_code != prev_code:
                        error_out = {"code": error_code, "inner_error": error_out}
                else:
                    error_out = {"code": error_code}
            except AttributeError:
                break

        error_out['message'] = self._exception_message
        if self._target is not None:
            error_out['target'] = self._target
        if self._details is not None:
            error_out['details'] = self._details

        # todo deprecate the following since AzureMLError will provide most of the necessary values for root error
        if self._message_format is not None:
            error_out['message_format'] = self._message_format
        if self._message_parameters is not None:
            error_out['message_parameters'] = json.dumps(self._message_parameters)
        if self._reference_code is not None:
            error_out['reference_code'] = self._reference_code

        if filter_fields:
            for key in filter_fields:
                error_out.pop(key, None)

        error_ret['error'] = error_out

        return json.dumps(error_ret, indent=indent)

    def _contains_code(self, code_hierarchy):
        """
        Determine whether this error exists within the hierarchy provided.
        :param code_hierarchy: List of strings sorted by hierarchy.
        :return: True if error exists in hierarchy; otherwise, False.
        :rtype: bool
        """
        error_response = json.loads(self._serialize_json())
        inner_error = error_response.get('error')
        response_hierarchy = []
        while inner_error is not None:
            response_hierarchy.append(inner_error.get('code'))
            inner_error = inner_error.get('inner_error')

        # Compare only the first K codes, allow more granular codes to be defined
        return code_hierarchy == response_hierarchy[:len(code_hierarchy)]

    def _get_error_hierarchy(self):
        return error_hierarchy.ErrorHierarchy(self._serialize_json())

    @staticmethod
    def _warn_deprecations(param, param_name):
        if param:
            warnings.warn("'{}' is deprecated. Please provide an instance of AzureMLError as a keyword argument "
                          "with key 'azureml_error'".format(param_name), DeprecationWarning)


class AzureMLAggregatedException(AzureMLException):
    """
    Initialize a new instance of AzureMLAggregatedException.

    :param exceptions_list: A list of messages describing the error.
    :type exceptions_list: builtin.list(str)
    """

    def __init__(self, exceptions_list, **kwargs):
        super(AzureMLAggregatedException, self).__init__("\n".join(exceptions_list), **kwargs)
