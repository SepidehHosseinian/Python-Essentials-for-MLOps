# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for creating a parameter to pass to a SQL stored procedure."""

from enum import Enum


class StoredProcedureParameter(object):
    """Represents a parameter passed to a SQL stored procedure.

    Use this class when defining a stored procedure in a :class:`azureml.data.sql_data_reference.SqlDataReference`
    object.

    :param name: The name of the stored procedure parameter.
    :type name: str
    :param value: The value of the stored procedure parameter.
    :type value: str
    :param type: The type of the stored procedure parameter value.
        The default is :class`azureml.data.stored_procedure_parameter.StoredProcedureParameterType`.String.
    :type type: azureml.data.stored_procedure_parameter.StoredProcedureParameterType
    """

    def __init__(self, name, value, type=None):
        """Class StoredProcedureParameter constructor.

        :param name: The name of the stored procedure parameter.
        :type name: str
        :param value: The value of the stored procedure parameter.
        :type value: str
        :param type: The type of the stored procedure parameter value.
            The default is azureml.data.stored_procedure_parameter.StoredProcedureParameterType.String.
        :type type: azureml.data.stored_procedure_parameter.StoredProcedureParameterType
        """
        self.name = name
        self.value = value
        self.type = type or StoredProcedureParameterType.String


class StoredProcedureParameterType(Enum):
    """Contains enumeration values describing the type of a stored procedure parameter.

    Use this class when defining a stored procedure parameter with
    the class :class:`azureml.data.stored_procedure_parameter.StoredProcedureParameter`.
    """

    String = "String"
    Int = "Int"
    Decimal = "Decimal"
    Guid = "Guid"
    Boolean = "Boolean"
    Date = "Date"
