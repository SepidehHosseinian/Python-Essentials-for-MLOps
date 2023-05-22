# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
Utility functions for Model Converters for Accel Models and IoT
"""
import json

from azureml.core.model import Model
from azureml.exceptions import WebserviceException, ModelNotFoundException


def _get_model_id(workspace, source_model):
    """
    Helper method to get model id.

    :param workspace:
    :type workspace: azureml.core.workspace.Workspace
    :param source_model:
    :type source_model: azureml.core.model or model id str
    :return: model id str
    :rtype: str
    """

    if type(source_model) is str:
        try:
            registered_model = Model(workspace, id=source_model)
        except WebserviceException:
            raise ModelNotFoundException('model not found')

        return registered_model.id

    if type(source_model) is Model:
        return source_model.id

    raise NotImplementedError('source_model must either be of type azureml.core.Model or a str of model id.')


def _get_as_str(input_value):
    """
    Helper method to serialize.

    :param input_value:
    :type input_value: str or List of str
    :return: serialized string
    :rtype: str
    """

    result = None
    if input_value:
        if type(input_value) == str:
            result = input_value
        elif type(input_value) == list:
            result = json.dumps(input_value)
        else:
            raise ValueError("Unexpected type [%s], str or list expected" % type(input_value))

    return result
