# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import re
import string
from typing import cast, Dict, Any

from azureml._common._error_definition.error_definition import ErrorDefinition


class _LogSafeFormatter(string.Formatter):
    """
    Custom string formatter to parse message parameters (inside ErrorDefinition -> message_format) that are log safe
    """
    def format_field(self, value, format_spec):
        if format_spec == ErrorDefinition.LOG_SAFE:
            return str(value)
        else:
            return "[MASKED]"


class AzureMLError:
    """
    Encapsulates all properties that are part of ErrorDefinition. It can be constructed by using the 'author time'
    properties (a.k.a ErrorDefinition) and optional 'runtime' properties such as message parameters, target and
    reference_code.

    An AzureMLException that contains an instance of this class can automatically allow the clients (e.g. UI) to
    localize any properties contained in the error; or provide Users with references to more context about the error,
    via properties such as error codes or a HTTP link to further drill down into how the error can be mitigated.

    Example usage:
    .. code-block:: python

                azureml_error = AzureMLError.create(SomeOtherErrorButSameErrorCode,
                                                    message_param_name="message_param_value",
                                                    target="target_for_the_error",
                                                    reference_code="a_reference_code")

                raise AzureMLException._with_error(azureml_error)
    """

    class Keys:
        """Constants to key into ErrorDefinition along with the runtime gathered parameters."""
        DETAILS_URI = "details_uri"
        CODE = "code"
        IS_TRANSIENT = "is_transient"
        INNER_ERROR = "inner_error"
        MESSAGE = "message"
        MESSAGE_FORMAT = "message_format"
        MESSAGE_PARAMETERS = "message_parameters"
        REFERENCE_CODE = "reference_code"
        TARGET = "target"

    def __init__(self, error_definition: ErrorDefinition, **kwargs):
        self._error_definition = error_definition
        self._target = kwargs.pop(AzureMLError.Keys.TARGET, None)
        self._reference_code = kwargs.pop(AzureMLError.Keys.REFERENCE_CODE, None)

        # Whatever's left are message_parameters to the error definition
        self._message_parameters = kwargs
        init_error = self._error_definition.validate(**self._message_parameters)
        if init_error is not None:
            raise KeyError(init_error)

        self._error_message = self._get_error_message()

    @staticmethod
    def create(
            cls: "type[ErrorDefinition]",
            **kwargs
    ) -> "AzureMLError":
        """
        Method to create an instance of this class, combining the static (dev-time) properties of the ErrorDefinition
        with any run-time ones available at the time of raising an error (e.g. target, reference_code)

        Example usage:
        .. code-block:: python

                    azureml_error = AzureMLError.create(SomeOtherErrorButSameErrorCode,
                                                        message_param_name="message_param_value",
                                                        target="target_for_the_error",
                                                        reference_code="a_reference_code")

        :param cls: An implementation of the ErrorDefinition class
        :param kwargs: Any run-time properties that the ErrorDefinition expects (such as reference_code)
        :return: An instance of AzureMLError.
        """
        error_definition = cast(ErrorDefinition, cls())
        azureml_error = AzureMLError(error_definition, **kwargs)
        return azureml_error

    @property
    def error_definition(self) -> ErrorDefinition:
        """The underlying error definition object."""
        return self._error_definition

    @property
    def error_message(self) -> str:
        """
        The error message obtained after all variable substitutions from message_parameters into the message_format.
        """
        return self._error_message

    @property
    def message_parameters(self) -> Dict[str, str]:
        """
        A dictionary of parameters to substitute within the message_format
        """
        return self._message_parameters

    @property
    def reference_code(self) -> str:
        """
        A helpful string that either a developer or the user can use to get further context on the error.
        """
        return self._reference_code

    @property
    def target(self) -> str:
        """
        The target to which the error applies.

        Example: For a BadArgumentError, this can point to the actual argument that caused the error.
        """
        return self._target

    def get_inner_errors(self) -> Dict[str, Any]:
        """
        Get all the inner errors defined by this error as a dictionary.

        Example:
            {
                "inner_error": {
                    "code": "UserError",
                    "inner_error": {
                        "code": "BadArgument",
                        "inner_error": {
                            "code": "ArgumentInvalid"
                        }
                    }
                }
            }
        """
        error_hierarchy = self.error_definition.code_hierarchy

        # Inner errors don't contain 'UserError' or 'System', so pop it out
        error_hierarchy.pop(0)

        result = inner_error_dict = {"code": error_hierarchy.pop(0)}

        for error in error_hierarchy:
            inner_error_dict["inner_error"] = dict()
            inner_error_dict = inner_error_dict["inner_error"]
            inner_error_dict["code"] = error

        return result

    def to_root_error_dict(self) -> Dict[str, str]:
        """Converts the error into a dictionary of keys that a RootError object expects."""
        return \
            {
                AzureMLError.Keys.CODE: self.error_definition.get_root_error(),
                AzureMLError.Keys.MESSAGE: self.error_message,
                AzureMLError.Keys.DETAILS_URI: self.error_definition.details_uri,
                AzureMLError.Keys.TARGET: self.target,
                AzureMLError.Keys.INNER_ERROR: self.get_inner_errors(),
                AzureMLError.Keys.MESSAGE_FORMAT: self.log_safe_message_format(),
                AzureMLError.Keys.MESSAGE_PARAMETERS: self.message_parameters,
                AzureMLError.Keys.REFERENCE_CODE: self.reference_code
            }

    def _get_error_message(self):
        """
        Populate the placeholders in the message_format by the message_parameters.
        Note: This can contain PII

        :return: A User understandable error message
        """
        if self._message_parameters:
            return self.error_definition.get_stripped_message_format().format(**self._message_parameters)

        # message_format itself is the error message
        return self.error_definition.message_format

    @staticmethod
    def _escape_message_parameters(message_format: str) -> str:
        """
        A helper method that escapes the parameters within message_format which are not safe to log.

        This is useful to prevent the str.format() method from populating the value of the PII parameters. This is done
        by wrapping those parameters within an additional set of curly braces

        e.g.
            Input: "A log safe param {non_pii_param:log_safe} v.s. a PII param {pii_param}."
            Output: "A log safe param {non_pii_param:log_safe} v.s. a PII param {{pii_param}}."

        :param message_format: The input string from which to escape the PII message parameters
        :return: The original message format with PII fields escaped
        """
        # regex for matching params inside '{XYZ}', but not '{{ABC}}' (since those are escaped)
        message_params_regex = re.compile(r"({[a-zA-Z_$0-9]*[:$.a-zA-Z_0-9]*?})(?!})")

        # find all message parameters declared within the message format
        message_params = message_params_regex.findall(message_format)

        # Filter all PII message parameters (i.e. those that don't end with ':log_safe}')
        log_safe_identifier_suffix = ":{}".format(ErrorDefinition.LOG_SAFE) + "}"
        non_log_safe_message_params = [k for k in message_params if not k.endswith(log_safe_identifier_suffix)]

        # For all the PII message parameters, escape it within '{}' (escape the already escaped braces)
        escaped_message_format = message_format.replace("{{", "{{{{").replace("}}", "}}}}")
        for cur_message_param in non_log_safe_message_params:
            escaped_message_format = escaped_message_format.replace(cur_message_param, '{' + cur_message_param + '}')

        return escaped_message_format

    def log_safe_message_format(self) -> str:
        """
        Formats the log safe (i.e. non-PII) fields within the error_definition.message_format

        e.g.
            Input: "A log safe param '{non_pii_param:log_safe}' v.s. a PII param {pii_param}."
            Output: "A log safe param 'that_can_be_logged' v.s. a PII param {pii_param}.

        :return: Message format for the error with log safe fields populated from the message parameters dictionary
        """
        escaped_message_format = self._escape_message_parameters(self.error_definition.message_format)
        fmt = _LogSafeFormatter()
        return fmt.format(escaped_message_format, **self.message_parameters)

    def __repr__(self):
        error_dict = {
            AzureMLError.Keys.CODE: self.error_definition.code,
            AzureMLError.Keys.MESSAGE: self.error_message,
            AzureMLError.Keys.INNER_ERROR: self.get_inner_errors(),
            AzureMLError.Keys.MESSAGE_FORMAT: self.log_safe_message_format(),
            AzureMLError.Keys.DETAILS_URI: self.error_definition.details_uri,
            AzureMLError.Keys.IS_TRANSIENT: self.error_definition.is_transient,
            AzureMLError.Keys.TARGET: self.target,
            AzureMLError.Keys.REFERENCE_CODE: self.reference_code
        }
        return json.dumps(error_dict, indent=4, sort_keys=True)

    def __str__(self):
        """
        Log safe representation of the error.

        Dev Note: Don't include PII fields in here.
        """
        error_dict = {
            AzureMLError.Keys.CODE: self.error_definition.code,
            AzureMLError.Keys.INNER_ERROR: self.get_inner_errors(),
            AzureMLError.Keys.MESSAGE_FORMAT: self.log_safe_message_format(),
            AzureMLError.Keys.DETAILS_URI: self.error_definition.details_uri,
            AzureMLError.Keys.IS_TRANSIENT: self.error_definition.is_transient,
            AzureMLError.Keys.TARGET: self.target,
            AzureMLError.Keys.REFERENCE_CODE: self.reference_code
        }
        return json.dumps(error_dict, indent=4, sort_keys=True)
