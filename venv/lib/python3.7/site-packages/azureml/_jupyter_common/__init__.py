# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Adding contrib methods to run for setting creation context."""
import os
import logging
from azureml.core import Experiment, Run
from azureml._restclient.constants import NOTEBOOK_ORIGIN
from azureml._restclient.models import CreateRunDto, CreatedFromDto
from azureml._file_utils.file_utils import normalize_path, traverse_up_path_and_find_file

NOTEBOOK_PATH_ENV_VARIABLE = 'AZUREML_NB_PATH'


def _set_created_from_content(self, type, path):
    """Set the run created from context."""
    if type == 'Notebook':
        artifact_name = os.path.basename(path)
        # Using the run-id as container,
        # because there is no caching behavior with the source notebook.
        artifact = self._client.artifacts.upload_artifact(
            path,
            origin=NOTEBOOK_ORIGIN,
            container=self.id,
            name=artifact_name)
        if artifact_name in artifact.artifacts:
            create_run_dto = CreateRunDto(run_id=self.id)
            create_run_dto.created_from = CreatedFromDto(
                type='Notebook',
                location_type='ArtifactId',
                location=artifact.artifacts[artifact_name].artifact_id)
            self._client.run_dto = self._client.patch_run(create_run_dto)
            return artifact
        else:
            raise RuntimeError("Couldn't upload the notebook artifact.")
    else:
        raise ValueError("Unsupported type.")


def _get_created_from_content(self):
    """Get a SAS to the creation context file or definition."""
    if (self._client.run_dto.created_from is not None
        and self._client.run_dto.created_from.type == "Notebook"
        and self._client.run_dto.created_from.location_type == "ArtifactId"
            and self._client.run_dto.created_from.location is not None):
        _, uri = self._client.artifacts.get_file_by_artifact_id(
            artifact_id=self._client.run_dto.created_from.location)
        return uri
    else:
        raise ValueError("Creation context does not exist or of unsupported type.")


Run.set_created_from_content = _set_created_from_content
Run.get_created_from_content = _get_created_from_content


def _update_run_created_from(run):
    try:
        if NOTEBOOK_PATH_ENV_VARIABLE in os.environ:
            notebook_path = traverse_up_path_and_find_file(
                path=normalize_path('.'),
                file_name=os.environ[NOTEBOOK_PATH_ENV_VARIABLE])
            # Best effort to save the current notebook.
            try:
                # Sending a comm message to the frontend to save the notebook.
                from ipykernel.comm import Comm
                import time
                target = Comm(target_name='azureml:save_notebook')
                target.send({
                    'notebookPath': os.environ[NOTEBOOK_PATH_ENV_VARIABLE]
                })
                # We wait for 3 seconds before snapshotting the notebook, so we get all the progress saved.
                time.sleep(3)
            except Exception:
                logging.warning("Notebook save comm message to jupyterlab failed.")

            if notebook_path != '':
                run.set_created_from_content('Notebook', notebook_path)
    except Exception:
        pass


def _experiment_submit_notebook_decorator(original_submit):
    def submit(self, config, tags=None, **kwargs):
        run = original_submit(self, config, tags, **kwargs)
        _update_run_created_from(run)
        return run
    return submit


def _experiment_start_logging_decorator(original_star_logging):
    def start_logging(self, *args, **kwargs):
        run = original_star_logging(self, *args, **kwargs)
        _update_run_created_from(run)
        return run
    return start_logging


Experiment.submit = _experiment_submit_notebook_decorator(Experiment.submit)
Experiment.start_logging = _experiment_start_logging_decorator(Experiment.start_logging)
