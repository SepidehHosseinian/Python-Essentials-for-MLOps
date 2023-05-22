# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains class that represents lineage information for datasets."""

import collections

from azureml.data._dataset import _Dataset


_Sequence = collections.abc.Sequence if hasattr(collections, 'abc') else collections.Sequence


class _InputDatasetsLineage(_Sequence):
    def __init__(self, workspace, input_datasets):
        self._workspace = workspace
        self._input_datasets = input_datasets
        self._lineage = None

    def __getitem__(self, key):
        self._initialize()
        return self._lineage[key]

    def __len__(self):
        self._initialize()
        return len(self._lineage)

    def __eq__(self, other):
        if isinstance(other, _InputDatasetsLineage):
            return str(self) == str(other)
        return False

    def __ne__(self, other):
        if not isinstance(other, _InputDatasetsLineage):
            return True
        return str(self) != str(other)

    def __str__(self):
        return str(self._resolve_lineage(False))

    def __repr__(self):
        return self.__str__()

    def _initialize(self):
        if self._lineage is None:
            self._lineage = self._resolve_lineage(True)

    def _resolve_lineage(self, instantiate):
        return [self._resolve_input_dataset(ds, instantiate) for ds in self._input_datasets]

    def _resolve_input_dataset(self, dataset_info, instantiate):
        saved_id = dataset_info['identifier'].get('savedId')
        resolved = {
            'dataset': {
                'id': saved_id
            },
            'consumptionDetails': {
                'type': dataset_info['consumptionType']
            }
        }
        if instantiate:
            resolved['dataset'] = _Dataset._get_by_id(self._workspace, saved_id)
        if 'inputDetails' in dataset_info.keys():
            resolved['consumptionDetails'].update(dataset_info['inputDetails'])
        return resolved


def update_output_lineage(workspace, output_datasets):
    if not output_datasets:
        return

    for value in output_datasets:
        value['dataset'] = _Dataset._get_by_id(workspace, value['identifier']['savedId'])
