# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Run script for dataset profile run."""
import sys
from azureml.core import Run, Dataset
from azureml.data._dataset_client import _DatasetClient


if __name__ == '__main__':
    dataset_id = sys.argv[1]
    action_id = sys.argv[2]
    saved_dataset_id = sys.argv[3]

    print('Start execution with action_id = {0}, dataset_id = {1} and '
          'saved_dataset_id = {2}'.format(action_id, dataset_id, saved_dataset_id))

    workspace = Run.get_context().experiment.workspace
    if workspace is None:
        raise TypeError('Workspace is found to be None')

    dataflow_json = None
    if saved_dataset_id:
        try:
            dataset = Dataset.get_by_id(workspace, saved_dataset_id)
        except Exception:
            errorMsg = 'Failed to get the dataset details by saved dataset id {}'.format(saved_dataset_id)
            print(errorMsg)
            raise TypeError(errorMsg)

    dataflow_json = dataset._dataflow.to_json()

    _DatasetClient._execute_dataset_action(workspace, dataset_id, action_id, dataflow_json)
