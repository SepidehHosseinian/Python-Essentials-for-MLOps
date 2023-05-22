# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A class for storing the field information."""


class _FieldInfo(object):
    """A class for storing the field information."""

    def __init__(self, field_type, documentation, list_element_type=None, user_keys=False, serialized_name=None,
                 exclude_if_none=False):
        """Class FieldInfo constructor.

        :param field_type: The data type of field.
        :type field_type: object
        :param documentation: The field information
        :type documentation: str
        :param list_element_type: The type of list element.
        :type list_element_type: object
        :param user_keys: user_keys=True, if keys in the value of the field are user keys.
                          user keys are not case normalized.
        :type user_keys: bool
        :param serialized_name:
        :type serialized_name: str
        :param exclude_if_none: Exclude from serialized output if value is None.
        :type exclude_if_none: bool
        """
        self._field_type = field_type
        self._documentation = documentation
        self._list_element_type = list_element_type
        self._user_keys = user_keys
        self._serialized_name = serialized_name
        self._exclude_if_none = exclude_if_none

    @property
    def field_type(self):
        """Get field type.

        :return: Returns the field type.
        :rtype: object
        """
        return self._field_type

    @property
    def documentation(self):
        """Return documentation.

        :return: Returns the documentation.
        :rtype: str
        """
        return self._documentation

    @property
    def list_element_type(self):
        """Get list element type.

        :return: Returns the list element type.
        :rtype: object
        """
        return self._list_element_type

    @property
    def user_keys(self):
        """Get user keys setting.

        :return: Returns the user keys setting.
        :rtype: bool
        """
        return self._user_keys

    @property
    def serialized_name(self):
        """Get serialized name.

        :return: Returns the serialized name.
        :rtype: str
        """
        return self._serialized_name

    @property
    def exclude_if_none(self):
        """Get whether to exclude None from serialized output.

        :return: Returns whether to exclude None form serialized output.
        :rtype: bool
        """
        return self._exclude_if_none
