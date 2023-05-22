# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json


class ModelConvertRequest(object):

    def __init__(self, modelId, sourceModelFlavor, targetModelFalvor,
                 toolName=None, toolVersion=None, modelConvertType=None):

        if toolVersion is None:
            toolVersion = "1.0"

        if modelConvertType is None:
            modelConvertType = "Generic"

        if toolName is None:
            raise ValueError("toolName is required")

        self.modelId = modelId
        self.sourceModelFlavor = sourceModelFlavor
        self.targetModelFlavor = targetModelFalvor
        self.toolName = toolName
        self.toolVersion = toolVersion
        self.modelConvertType = modelConvertType
        self.compilationOptions = {}

    @property
    def model_id(self):
        return self.modelId

    @model_id.setter
    def model_id(self, value):
        self.modelId = value

    @property
    def source_model_flavor(self):
        return self.sourceModelFlavor

    @source_model_flavor.setter
    def source_model_flavor(self, value):
        self.sourceModelFlavor = value

    @property
    def target_model_flavor(self):
        return self.targetModelFlavor

    @target_model_flavor.setter
    def target_model_flavor(self, value):
        self.targetModelFlavor = value

    @property
    def model_convert_type(self):
        return self.modelConvertType

    @model_convert_type.setter
    def model_convert_type(self, value):
        self.modelConvertType = value

    @property
    def tool_name(self):
        return self.toolName

    @tool_name.setter
    def tool_name(self, value):
        self.toolName = value

    @property
    def tool_version(self):
        return self.toolVersion

    @tool_version.setter
    def tool_version(self, value):
        self.toolVersion = value

    @property
    def compilation_options(self):
        return self.compilationOptions

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
