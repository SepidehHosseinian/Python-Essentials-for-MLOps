# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from collections import OrderedDict
import errno
import logging
import os
import time

from six.moves.queue import Queue, Empty

from azureml._html.utilities import to_html, make_link
from azureml._logging import ChainedIdentity
from azureml._restclient.constants import RUN_ORIGIN, RUN_USER_AGENT
from azureml._restclient.models.run_type_v2 import RunTypeV2
from azureml._run_impl.run_context_manager import RunContextManager

from azureml.core._portal import HasRunPortal
from azureml.core._docs import get_docs_url
from azureml.history._tracking import get_py_wd

HEARTBEAT_INTERVAL = 120

module_logger = logging.getLogger(__name__)


# TODO: Can we get rid of this class soon? Why does it exist?
class _RunBase(ChainedIdentity, HasRunPortal):

    _registered_kill_handlers = Queue()

    def __init__(self, experiment, run_id, outputs=None, logs=None,
                 _run_dto=None, _worker_pool=None, _user_agent=None, _ident=None,
                 _batch_upload_metrics=True, py_wd=None, deny_list=None,
                 flush_eager=False, redirect_output_stream=True, **kwargs):
        """
        :param experiment: The experiment.
        :type experiment: azureml.core.experiment.Experiment
        :param run_id: The run id for the run.
        :type run_id: str
        :param outputs: The outputs to be tracked
        :type outputs:
        :param logs: The logs directory to be tracked
        :type logs:
        :param _worker_pool: The worker pool for async tasks
        :type _worker_pool: azureml._async.worker_pool.WorkerPool
        :param clean_up: If true, call _register_kill_handler from run_base
        :type clean_up: bool
        """
        # _worker_pool needed for backwards compat

        from azureml._run_impl.run_history_facade import RunHistoryFacade

        self._experiment = experiment

        self._run_id = run_id

        _ident = _ident if _ident is not None else ChainedIdentity.DELIM.join([self.__class__.__name__, self._run_id])

        run_clean_up = kwargs.pop("clean_up", True)

        # We need to do this in order to resolve the history object :(
        # Get rid of just name and pass the objects around BUT
        # TODO: Everything needs to use the *SAME PARAMETER NAMES*
        super(_RunBase, self).__init__(
            experiment=self._experiment,
            run_id=self._run_id,
            _ident=_ident,
            **kwargs)

        user_agent = _user_agent if _user_agent is not None else RUN_USER_AGENT

        # Create an outputs directory if one does not exist
        if outputs is not None:
            outputs = [outputs] if isinstance(outputs, str) else outputs
        else:
            outputs = []

        for output in outputs:
            try:
                os.makedirs(output)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        py_wd = get_py_wd() if py_wd is None else py_wd

        self._client = RunHistoryFacade(self._experiment, self._run_id, RUN_ORIGIN, run_dto=_run_dto,
                                        worker_pool=self._experiment.workspace.service_context.worker_pool,
                                        outputs=outputs, py_wd=py_wd, deny_list=deny_list,
                                        user_agent=user_agent, _parent_logger=self._logger,
                                        _batch_upload_metrics=_batch_upload_metrics, flush_eager=flush_eager)

        # self._run_dto property does some time-expensive serialization
        # so just materialize it once for use to populate all other fields
        _run_dto_as_dict = self._run_dto

        self._root_run_id = _run_dto_as_dict["root_run_id"]
        self._outputs = outputs
        self._run_number = _string_to_int(_run_dto_as_dict["run_number"], "run number")
        self._run_source = _run_dto_as_dict.get("properties", {}).get("azureml.runsource", None)
        self._runtype = _run_dto_as_dict.get("run_type", self._run_source)
        self._run_name = _run_dto_as_dict.get("name", None)
        self._logger.debug("Constructing run from dto. type: %s, source: %s, props: %s",
                           self._runtype,
                           self._run_source,
                           _run_dto_as_dict.get("properties", {}))
        run_type_v2 = _run_dto_as_dict.get("run_type_v2", None)
        if run_type_v2:
            self._runtype_v2 = RunTypeV2(orchestrator=run_type_v2.get("orchestrator", None),
                                         traits=run_type_v2.get("traits", None))
        self._context_manager = RunContextManager(self, logs=logs,
                                                  heartbeat_enabled=_run_dto_as_dict.get("heartbeat_enabled", False),
                                                  _parent_logger=self._logger, py_wd=py_wd,
                                                  redirect_output_stream=redirect_output_stream)

        if run_clean_up:
            self._register_kill_handler(self._cleanup)

    @classmethod
    def get_docs_url(cls):
        return get_docs_url(cls)

    @property
    def _run_dto(self):
        """Return the internal representation of a run."""

        run_dto = self._client.run_dto

        if isinstance(run_dto, dict):
            self._logger.debug("Return run dto as existing dict")
            return run_dto
        else:
            return self._client.run.dto_to_dictionary(run_dto)

    def _get_base_info_dict(self):
        return OrderedDict([
            ('Experiment', self._experiment.name),
            ('Id', self._run_id),
            ('Type', self._runtype),
            ('Status', self._client.run_dto.status)
        ])

    def __str__(self):
        info = self._get_base_info_dict()
        formatted_info = ',\n'.join(["{}: {}".format(k, v) for k, v in info.items()])
        return "Run({0})".format(formatted_info)

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        info = self._get_base_info_dict()
        info.update([
            ('Details Page', make_link(self.get_portal_url(), "Link to Azure Machine Learning studio")),
            ('Docs Page', make_link(self.get_docs_url(), "Link to Documentation"))
        ])
        return to_html(info)

    def __enter__(self):
        return self._context_manager.__enter__()

    def __exit__(self, exit_type, value, traceback):
        return self._context_manager.__exit__(exit_type, value, traceback)

    def _heartbeat(self):
        self._client.run.post_event_heartbeat(HEARTBEAT_INTERVAL)

    @classmethod
    def _kill(cls, timeout=40):
        print("Cleaning up all outstanding Run operations, waiting {} seconds".format(timeout))
        handlers = []
        while True:
            try:
                handlers.append(cls._registered_kill_handlers.get_nowait())
            except Empty:
                break

        print("{} items cleaning up...".format(len(handlers)))
        start_time = time.time()
        end_time = start_time + timeout
        for handler in handlers:
            if time.time() > end_time:
                module_logger.warn("Could not clean up all items! Data loss might occur!")
                return
            handler(timeout)

        print("Cleanup took {} seconds".format(time.time() - start_time))

    @classmethod
    def _register_kill_handler(cls, work):
        cls._registered_kill_handlers.put(work)

    def _cleanup(self, timeout):
        # TODO: Structure this better once we know of more cases
        self._client.flush(timeout)


def _string_to_int(value, details):
    try:
        return int(value)
    except ValueError as ve:
        raise ValueError("Could not convert [{0}] to int for {1}: {2}".format(value, details, ve))
