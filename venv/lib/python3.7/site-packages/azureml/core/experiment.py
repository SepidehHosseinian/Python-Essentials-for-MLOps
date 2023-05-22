# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality used to submit experiments and manage experiment history in Azure Machine Learning."""

import logging
import os
from collections import OrderedDict
from functools import wraps

from azureml._logging import ChainedIdentity
from azureml._project import _commands
from azureml._base_sdk_common.common import check_valid_resource_name
from azureml._restclient.workspace_client import WorkspaceClient

from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.core._experiment_method import get_experiment_submit, check_for_lock_file
from azureml.core._portal import HasExperimentPortal
from azureml.core._docs import get_docs_url
from azureml.exceptions import UserErrorException
from azureml._html.utilities import to_html, make_link
from azureml.core.compute import ComputeInstance
from datetime import datetime, date

module_logger = logging.getLogger(__name__)


def _check_for_experiment_id(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if self._id is None:
            raise UserErrorException("{} doesn't have an id set therefore, the {} cannot "
                                     "modify the experiment. Please call the Experiment "
                                     "constructor by setting _create_in_cloud to True".format(
                                         self, self.__class__.__name__))
        return func(self, *args, **kwargs)
    return wrapped


class ViewType(object):
    """Defines how to filter out or include archived experiments when listing experiments.

    Use ViewType in the :meth:`azureml.core.experiment.Experiment.list` method.

    Attributes:
        ActiveOnly: Show only active experiments.

        All: Show all experiments.

        ArchivedOnly: Show only archived experiments.

    """

    ActiveOnly = "ActiveOnly"
    All = "All"
    ArchivedOnly = "ArchivedOnly"


class Experiment(ChainedIdentity, HasExperimentPortal):
    """Represents the main entry point for creating and working with experiments in Azure Machine Learning.

    An Experiment is a container of *trials* that represent multiple model runs.

    .. remarks::

        An Azure Machine Learning *experiment* represent the collection of *trials* used to validate a
        user's *hypothesis*.

        In Azure Machine Learning, an experiment is represented by the :class:`azureml.core.Experiment`
        class and a trial is represented by the :class:`azureml.core.Run` class.

        To get or create an experiment from a workspace, you request the experiment using the experiment name.
        Experiment name must be 3-36 characters, start with a letter or a number, and can only contain letters,
        numbers, underscores, and dashes.

        .. code-block:: python

            experiment = Experiment(workspace, "MyExperiment")

        If the experiment is not found in the workspace, a new experiment is created.

        There are two ways to execute an experiment trial. If you are interactively experimenting in a
        Jupyter Notebook, use  :func:`start_logging` If you are submitting an experiment from source code
        or some other type of configured trial, use :func:`submit`

        Both mechanisms create a :class:`azureml.core.Run` object.  In interactive scenarios, use logging methods
        such as :func:`azureml.core.Run.log` to add measurements and metrics to the trial record.  In configured
        scenarios use status methods such as :func:`azureml.core.Run.get_status` to retrieve information about the
        run.

        In both cases you can use query methods like :func:`azureml.core.Run.get_metrics` to retrieve the current
        values, if any, of any trial measurements and metrics.


    :param workspace: The workspace object containing the experiment.
    :type workspace: azureml.core.workspace.Workspace
    :param name: The experiment name.
    :type name: str
    :param kwargs: A dictionary of keyword args.
    :type kwargs: dict
    """

    def __init__(self,
                 workspace,
                 name,
                 _skip_name_validation=False,
                 _id=None,
                 _archived_time=None,
                 _create_in_cloud=True,
                 _experiment_dto=None,
                 **kwargs):
        """Experiment constructor.

        :param workspace: The workspace object containing the experiment.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The experiment name.
        :type name: str
        :param kwargs: A dictionary of keyword args.
        :type kwargs: dict
        """
        self._workspace = workspace
        self._name = name
        self._workspace_client = WorkspaceClient(workspace.service_context)

        _ident = kwargs.pop("_ident", ChainedIdentity.DELIM.join([self.__class__.__name__, self._name]))

        if not _skip_name_validation:
            check_valid_resource_name(name, "Experiment")

        # Get or create the experiment from the workspace
        if _create_in_cloud:
            experiment = self._workspace_client.get_or_create_experiment(experiment_name=name)
            self._id = experiment.experiment_id
            self._archived_time = experiment.archived_time
            self._extract_from_dto(experiment)
        else:
            self._id = _id
            self._archived_time = _archived_time
            self._extract_from_dto(_experiment_dto)

        super(Experiment, self).__init__(experiment=self, _ident=_ident, **kwargs)

    def submit(self, config, tags=None, **kwargs):
        r"""Submit an experiment and return the active created run.

        .. remarks::

            Submit is an asynchronous call to the Azure Machine Learning platform to execute a trial on local
            or remote hardware.  Depending on the configuration, submit will automatically prepare
            your execution environments, execute your code, and capture your source code and results
            into the experiment's run history.

            To submit an experiment you first need to create a configuration object describing
            how the experiment is to be run.  The configuration depends on the type of trial required.

            An example of how to submit an experiment from your local machine is as follows:

            .. code-block:: python

                from azureml.core import ScriptRunConfig

                # run a trial from the train.py code in your current directory
                config = ScriptRunConfig(source_directory='.', script='train.py',
                    run_config=RunConfiguration())
                run = experiment.submit(config)

                # get the url to view the progress of the experiment and then wait
                # until the trial is complete
                print(run.get_portal_url())
                run.wait_for_completion()

            For details on how to configure a run, see the configuration type details.

            * :class:`azureml.core.ScriptRunConfig`
            * :class:`azureml.train.automl.automlconfig.AutoMLConfig`
            * :class:`azureml.pipeline.core.Pipeline`
            * :class:`azureml.pipeline.core.PublishedPipeline`
            * :class:`azureml.pipeline.core.PipelineEndpoint`

            .. note::

                When you submit the training run, a snapshot of the directory that contains your training scripts \
                is created and sent to the compute target. It is also stored as part of the experiment in your \
                workspace. If you change files and submit the run again, only the changed files will be uploaded.

            To prevent files from being included in the snapshot, create a
            `.gitignore <https://git-scm.com/docs/gitignore>`_ or `.amlignore` file in the directory and add the
            files to it. The `.amlignore` file uses the same syntax and patterns as the .gitignore file. If both
            files exist, the `.amlignore` file takes precedence.

            For more information, see `Snapshots
            <https://docs.microsoft.com/azure/machine-learning/concept-azure-machine-learning-architecture#snapshots>`_.

        :param config: The config to be submitted.
        :type config: object
        :param tags: Tags to be added to the submitted run, {"tag": "value"}.
        :type tags: dict
        :param kwargs: Additional parameters used in submit function for configurations.
        :type kwargs: dict
        :return: A run.
        :rtype: azureml.core.Run
        """
        # Warn user if trying to run GPU image on a local machine
        try:
            runconfig = config.run_config
            if (runconfig.environment.docker.base_image == DEFAULT_GPU_IMAGE and runconfig.target == "local"):
                print("Note: The GPU base image must be used on Microsoft Azure Services only. See LICENSE.txt file.")
        except AttributeError:  # Not all configuration options have run_configs that specify base images
            pass

        # Warn user if conda lock files haven't been removed for non-Docker run
        try:
            runconfig = config.run_config
            if not runconfig.environment.docker.enabled:
                check_for_lock_file()
        except AttributeError:  # Not all configuration options have run_configs that specify environments
            pass

        try:
            # Warn user if compute instance was created before 19/9/2021 and if it was not restarted after that
            if config.run_config.target != 'local':
                instance = ComputeInstance(self.workspace, config.run_config.target)
                creation_time = instance.status.serialize().get('creationTime', None)
                instance_state = instance.status.serialize().get('state', None)
                if creation_time is not None:
                    creation_date = datetime.fromisoformat(creation_time).date()
                if instance.type == 'ComputeInstance' and instance_state.lower() != 'stopped'\
                        and creation_date < date(2021, 9, 19):
                    self._logger.warning("The internal settings on your compute instance are out of date and require a"
                                         "restart before May 31, 2022. Some features will be interrupted after the"
                                         "May 31, 2022 until a restart is performed.")
        except Exception:
            pass

        submit_func = get_experiment_submit(config)
        with self._log_context("submit config {}".format(config.__class__.__name__)):
            run = submit_func(config, self.workspace, self.name, **kwargs)
        if tags is not None:
            run.set_tags(tags)
        return run

    def start_logging(self, *args, **kwargs):
        """Start an interactive logging session and create an interactive run in the specified experiment.

        .. remarks::

            **start_logging** creates an interactive run for use in scenarios such as Jupyter Notebooks.
            Any metrics that are logged during the session are added to the run record in the experiment.
            If an output directory is specified, the contents of that directory is uploaded as run
            artifacts upon run completion.

            .. code-block:: python

                experiment = Experiment(workspace, "My Experiment")
                run = experiment.start_logging(outputs=None, snapshot_directory=".", display_name="My Run")
                ...
                run.log_metric("Accuracy", accuracy)
                run.complete()

            .. note::

                **run_id** is automatically generated for each run and is unique within the experiment.

        :param experiment: The experiment.
        :type experiment: azureml.core.Experiment
        :param outputs: Optional outputs directory to track. For no outputs, pass False.
        :type outputs: str
        :param snapshot_directory: Optional directory to take snapshot of. Setting to None will take no snapshot.
        :type snapshot_directory: str
        :param args:
        :type args: builtin.list
        :param kwargs:
        :type kwargs: dict
        :return: Return a started run.
        :rtype: azureml.core.Run
        """
        from azureml.core.run import Run
        return Run._start_logging(self, *args, _parent_logger=self._logger, **kwargs)

    @_check_for_experiment_id
    def archive(self):
        """Archive an experiment.

        .. remarks::

            After archival, the experiment will not be listed by default.
            Attempting to write to an archived experiment will create a new active experiment with the same
            name.
            An archived experiment can be restored by calling :func:`azureml.core.Experiment.reactivate` as long as
            there is not another
            active experiment with the same name.
        """
        updated_experiment = self._workspace_client.archive_experiment(experiment_id=self._id)
        self._archived_time = updated_experiment.archived_time

    @_check_for_experiment_id
    def reactivate(self, new_name=None):
        """Reactivates an archived experiment.

        .. remarks::

            An archived experiment can only be reactivated if there is not another active experiment with
            the same name.

        :param new_name: Not supported anymore
        :type new_name: str
        """
        if new_name is not None:
            raise UserErrorException("Cannot rename an experiment. If the archived experiment name conflicts"
                                     " with an active experiment name, you can delete the active experiment"
                                     " before unarchiving this experiment.")
        updated_experiment = self._workspace_client.reactivate_experiment(experiment_id=self._id)
        self._archived_time = updated_experiment.archived_time
        self._name = updated_experiment.name

    @_check_for_experiment_id
    def tag(self, key, value=None):
        """Tag the experiment with a string key and optional string value.

        .. remarks::

            Tags on an experiment are stored in a dictionary with string keys and string values.
            Tags can be set, updated and deleted.
            Tags are user-facing and generally contain meaning information for the consumers of the experiment.

            .. code-block:: python

                experiment.tag('')
                experiment.tag('DeploymentCandidate')
                experiment.tag('modifiedBy', 'Master CI')
                experiment.tag('modifiedBy', 'release pipeline') # Careful, tags are mutable


        :param key: The tag key
        :type key: str
        :param value: An optional value for the tag
        :type value: str
        """
        self.set_tags({key: value})

    @_check_for_experiment_id
    def set_tags(self, tags):
        """Add or modify a set of tags on the experiment. Tags not passed in the dictionary are left untouched.

        :param tags: The tags stored in the experiment object
        :type tags: dict[str]
        """
        self._workspace_client.set_tags(experiment_id=self._id, tags=tags)

    @_check_for_experiment_id
    def remove_tags(self, tags):
        """Delete the specified tags from the experiment.

        :param tags: The tag keys that will get removed
        :type tags: [str]
        """
        self._workspace_client.delete_experiment_tags(experiment_id=self._id, tags=tags)

    @_check_for_experiment_id
    def refresh(self):
        """Return the most recent version of the experiment from the cloud."""
        experiment = self._workspace_client.get_experiment_by_id(self._id)
        self._archived_time = experiment.archived_time
        self._extract_from_dto(experiment)

    @staticmethod
    def from_directory(path, auth=None):
        """(Deprecated) Load an experiment from the specified path.

        :param path: Directory containing the experiment configuration files.
        :type path: str
        :param auth: The auth object.
            If None the default Azure CLI credentials will be used or the API will prompt for credentials.
        :type auth: azureml.core.authentication.ServicePrincipalAuthentication or
            azureml.core.authentication.InteractiveLoginAuthentication
        :return: Returns the Experiment
        :rtype: azureml.core.Experiment
        """
        from azureml.core.workspace import Workspace

        info_dict = _commands.get_project_info(auth, path)

        # TODO: Fix this
        subscription = info_dict[_commands.SUBSCRIPTION_KEY]
        resource_group_name = info_dict[_commands.RESOURCE_GROUP_KEY]
        workspace_name = info_dict[_commands.WORKSPACE_KEY]
        experiment_name = info_dict[_commands.PROJECT_KEY]

        workspace = Workspace(
            subscription_id=subscription, resource_group=resource_group_name, workspace_name=workspace_name, auth=auth
        )
        return Experiment(workspace=workspace, name=experiment_name)

    @staticmethod
    def list(workspace, experiment_name=None, view_type=ViewType.ActiveOnly, tags=None):
        """Return the list of experiments in the workspace.

        :param workspace: The workspace from which to list the experiments.
        :type workspace: azureml.core.workspace.Workspace
        :param experiment_name: Optional name to filter experiments.
        :type experiment_name: str
        :param view_type: Optional enum value to filter or include archived experiments.
        :type view_type: azureml.core.experiment.ViewType
        :param tags: Optional tag key or dictionary of tag key-value pairs to filter experiments on.
        :type tag: str or dict[str]
        :return: A list of experiment objects.
        :rtype: builtin.list[azureml.core.Experiment]
        """
        workspace_client = WorkspaceClient(workspace.service_context)
        experiments = workspace_client.list_experiments(experiment_name=experiment_name,
                                                        view_type=view_type, tags=tags)
        return [Experiment(workspace,
                           experiment.name,
                           _id=experiment.experiment_id,
                           _archived_time=experiment.archived_time,
                           _create_in_cloud=False,
                           _skip_name_validation=True,
                           _experiment_dto=experiment) for experiment in experiments]

    @staticmethod
    def delete(workspace, experiment_id):
        """Delete an experiment in the workspace.

        :param workspace: The workspace which the experiment belongs to.
        :type workspace: azureml.core.workspace.Workspace
        :param experiment_id: The experiment id of the experiment to be deleted.
        :type experiment_name: str
        """
        workspace_client = WorkspaceClient(workspace.service_context)
        delete_experiment_timeout_seconds = int(os.getenv('AZUREML_DELETE_EXPERIMENT_TIMEOUT_SECONDS', 240))
        workspace_client.delete_experiment(experiment_id, timeout_seconds=delete_experiment_timeout_seconds)

    @property
    def workspace(self):
        """Return the workspace containing the experiment.

        :return: Returns the workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        return self._workspace

    @property
    def workspace_object(self):
        """(Deprecated) Return the workspace containing the experiment.

        Use the :attr:`azureml.core.experiment.Experiment.workspace` attribute.

        :return: The workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        self._logger.warning("Deprecated, use experiment.workspace")
        return self.workspace

    @property
    def name(self):
        """Return name of the experiment.

        :return: The name of the experiment.
        :rtype: str
        """
        return self._name

    @property
    def id(self):
        """Return id of the experiment.

        :return: The id of the experiment.
        :rtype: str
        """
        return self._id

    @property
    def archived_time(self):
        """Return the archived time for the experiment. Value should be None for an active experiment.

        :return: The archived time of the experiment.
        :rtype: str
        """
        return self._archived_time

    @property
    def tags(self):
        """Return the mutable set of tags on the experiment.

        :return: The tags on an experiment.
        :rtype: dict[str]
        """
        return self._tags

    def get_runs(self, type=None, tags=None, properties=None, include_children=False):
        """Return a generator of the runs for this experiment, in reverse chronological order.

        :param type: Filter the returned generator of runs by the provided type. See
            :func:`azureml.core.Run.add_type_provider` for creating run types.
        :type type: string
        :param tags: Filter runs by "tag" or {"tag": "value"}.
        :type tags: string or dict
        :param properties: Filter runs by "property" or {"property": "value"}
        :type properties: string or dict
        :param include_children: By default, fetch only top-level runs. Set to true to list all runs.
        :type include_children: bool
        :return: The list of runs matching supplied filters.
        :rtype: builtin.list[azureml.core.Run]
        """
        from azureml.core.run import Run
        return Run.list(self, type=type, tags=tags, properties=properties, include_children=include_children)

    def _serialize_to_dict(self):
        """Serialize the Experiment object details into a dictionary.

        :return: experiment details
        :rtype: dict
        """
        output_dict = {"Experiment name": self.name,
                       "Subscription id": self.workspace.subscription_id,
                       "Resource group": self.workspace.resource_group,
                       "Workspace name": self.workspace.name}
        return output_dict

    def _get_base_info_dict(self):
        """Return base info dictionary.

        :return:
        :rtype: OrderedDict
        """
        return OrderedDict([
            ('Name', self._name),
            ('Workspace', self._workspace.name)
        ])

    @classmethod
    def get_docs_url(cls):
        """Url to the documentation for this class.

        :return: url
        :rtype: str
        """
        return get_docs_url(cls)

    def __str__(self):
        """Format Experiment data into a string.

        :return:
        :rtype: str
        """
        info = self._get_base_info_dict()
        formatted_info = ',\n'.join(["{}: {}".format(k, v) for k, v in info.items()])
        return "Experiment({0})".format(formatted_info)

    def __repr__(self):
        """Representation of the object.

        :return: Return the string form of the experiment object
        :rtype: str
        """
        return self.__str__()

    def _repr_html_(self):
        """Html representation of the object.

        :return: Return an html table representing this experiment
        :rtype: str
        """
        info = self._get_base_info_dict()
        info.update([
            ('Report Page', make_link(self.get_portal_url(), "Link to Azure Machine Learning studio")),
            ('Docs Page', make_link(self.get_docs_url(), "Link to Documentation"))
        ])
        return to_html(info)

    def _extract_from_dto(self, experiment_dto):
        if experiment_dto is None:
            self._experiment_dto = experiment_dto
            self._tags = None
        else:
            self._experiment_dto = experiment_dto
            self._tags = experiment_dto.tags
