# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Abstract Dataset class. (re-export)"""

from azureml.data.abstract_dataset import AbstractDataset, _DatasetRegistration, _DatasetTelemetryInfo, \
    _DatasetDict, _DatasetDictKeyIterator, _get_path_from_step

_Dataset = AbstractDataset
_DatasetRegistration = _DatasetRegistration
_DatasetTelemetryInfo = _DatasetTelemetryInfo
_DatasetDict = _DatasetDict
_DatasetDictKeyIterator = _DatasetDictKeyIterator
_get_path_from_step = _get_path_from_step
