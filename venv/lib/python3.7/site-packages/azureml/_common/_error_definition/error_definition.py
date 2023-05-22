# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
from abc import ABC, abstractmethod
from typing import Optional, List


class ErrorDefinition(ABC):
    """
    ErrorDefinition is an abstraction for error properties that are known at authoring time (dev time).
    A new error is authored by simply providing an implementation for this abstraction.
    This must be done be identifying the category of the error and then deriving from a root error that
    represents that category. This 'author time' properties ensure that an instance of this can be safely logged,
    because it impossible for it to contain PII
    """

    # A format specifier for message parameters inside the message_format indicating the parameter is a non-PII field.
    # This can be used as a regular python format specifier (e.g. "Sample string {param:log_safe}")
    LOG_SAFE = "log_safe"

    @property
    @abstractmethod
    def message_format(self) -> str:
        """
        An un-formatted version of the message with no variable substitution. The string can contain any
        number of fields surrounded by a single curly brace, but it is allowed to contain zero such fields.
        Field identifiers within the curly braces should not contain whitespace or special characters.
        The MessageFormat can never contain any customer content.

        Example:
            "Entity with ID {entity_id} and name {entity_name} cannot be deleted."

        Example2: (with a non-PII indicator for one of the message parameters)
            "Entity with ID {entity_id:log_safe} and name {entity_name} cannot be deleted."
        """
        raise NotImplementedError

    @property
    def details_uri(self) -> Optional[str]:
        """
        A URI which points to more details about the context of the error.
        The value is intended to be consumed by interactive clients to provide links (either deep into the same
        client, or external) to pages detailing the error and remediation or logs that contain more information.
        """
        return None

    @property
    def is_transient(self) -> bool:
        """
        Indicates whether the error is transient (and possibly retry-able by the receiver of this error)
        """
        return False

    @property
    def use_parent_error_code(self) -> bool:
        """
        Property to indicate that this class only extends from the parent for the sake of overriding the
        message_format, but wants to keep the same error code as the parent class.

        Dev Note: If this is being overridden by a class, it indicates that the class is a terminal (leaf) node in the
        error hierarchy (as this class won't be featuring in the error code / hierarchy).
        """
        return False

    @property
    def code(self) -> str:
        """
        The error code represented by this class. By default, it is the name of the class.

        Dev Note: This property shouldn't be overridden by the inheritors. Instead, define a new implementation of
        ErrorDefinition
        """
        return self._get_error_hierarchy()[-1]

    @property
    def code_hierarchy(self) -> List[str]:
        """
        The entire hierarchy of the error code.
        Example: [UserError, BadArgument, ArgumentOutOfRange]
       """
        return self._get_error_hierarchy()

    def get_stripped_message_format(self) -> str:
        """
        Removes all occurrences of the [custom] format specifiers within the message parameters of message format.
        If there were no format specifiers attached to the message format, the returned value is the same as
        the original message format.

        :return: message_format without any message_parameter indicator fields
        """
        log_safe_indicator = ":" + ErrorDefinition.LOG_SAFE
        stripped_message_format = self.message_format.replace(log_safe_indicator, "")
        return stripped_message_format

    def get_root_error(self) -> str:
        """
        Get the top most error code represented by this object.

        :return: 'UserError' or 'SystemError'
        """
        error_hierarchy = self._get_error_hierarchy()
        return error_hierarchy[0] if error_hierarchy else 'SystemError'

    def validate(self, **message_parameters) -> Optional[str]:
        """
        A helper method to validate that the message format contained in this class can be safely substituted with
        the provided message parameters.

        :param message_parameters: The keyword arguments to substitute in the message_format
        :return: None, if the validation succeeds
        :raises KeyError if there is a substitution defined in the message format but without an equivalent
        message parameter provided
        """
        try:
            self.get_stripped_message_format().format(**message_parameters)
        except KeyError as ke:
            return (
                "Message parameters contains missing keys required for message format. "
                "Missing Key: {}".format(str(ke))
            )

        return None

    def _get_error_hierarchy(self) -> List[str]:
        error_hierarchy = []  # type: List[str]
        all_super_classes = [c.__name__ for c in self.__class__.__mro__]

        if self.use_parent_error_code:
            all_super_classes.pop(0)

        for error_code in all_super_classes:
            if error_code == "ErrorDefinition":
                break
            error_hierarchy.append(error_code)

        return list(reversed(error_hierarchy))

    def __str__(self):
        error_dict = {
            "code": self.code,
            "code_hierarchy": self._get_error_hierarchy(),
            "message_format": self.message_format,
            "details_uri": self.details_uri,
            "is_transient": self.is_transient,
        }
        return json.dumps(error_dict, indent=4, sort_keys=True)

    def __repr__(self):
        return self.__str__()
