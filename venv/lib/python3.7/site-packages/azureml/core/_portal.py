# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token

_PUBLIC_PORTAL_DOMAIN_NAME = 'https://ml.azure.com'


def _warn_on_preview_url(url):
    # type: (str) -> bool
    if "azureml-test.net" in url or "history" in url:
        logging.warn("URL is not final public url")


class HasPortal(object):
    """ Mixin protocol providing Azure Portal links for workspaces """

    def get_portal_url(self):
        raise NotImplementedError("No portal URL implemented")


class HasWorkspacePortal(HasPortal):
    """
    Mixin for providing Azure Portal links for workspaces.
    :param workspace:
    :type workspace: Workspace
    """

    WORKSPACE_FMT = '?wsid=/subscriptions/{0}' \
                    '/resourcegroups/{1}' \
                    '/workspaces/{2}'

    TID_FMT = "&tid={0}"

    def __init__(self, workspace):
        """ The constructor method """

        self._portal_url = HasWorkspacePortal._get_portal_domain(workspace)

        auth = workspace._auth_object
        try:
            self._formatted_tid = HasWorkspacePortal.TID_FMT.format(
                fetch_tenantid_from_aad_token(auth._get_arm_token()))
        except Exception:
            # If we are unable to get tid from token, proceed without it.
            self._formatted_tid = ""

        self._workspace_url = \
            self._portal_url + \
            HasWorkspacePortal.WORKSPACE_FMT.format(
                workspace.subscription_id,
                workspace.resource_group,
                workspace.name) + \
            self._formatted_tid

    @property
    def portal_url(self):
        return self._portal_url

    @staticmethod
    def _get_portal_domain(workspace):
        """Return domain for portal based on cloud type.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :return:
        :rtype: str
        """
        portal_domain = workspace.service_context._get_workspace_portal_url()
        if not portal_domain:
            logging.warn("Fail to fetch workspace portal url, use public azure portal as default.")
            portal_domain = _PUBLIC_PORTAL_DOMAIN_NAME

        return portal_domain

    def get_portal_url(self):
        """
        :return: Returns the Azure portal url for the workspace.
        :rtype: str
        """
        # type () -> str
        _warn_on_preview_url(self._workspace_url)
        return self._workspace_url


class HasExperimentPortal(HasWorkspacePortal):
    """
    Mixin for providing experiment links to the Azure portal.
    :param experiment:
    :type experiment: Experiment
    """

    EXPERIMENT_NAME_PATH = '/experiments/{0}'
    EXPERIMENT_ID_PATH = '/experiments/id/{0}'

    def __init__(self, experiment):
        """ The constructor method """
        super(HasExperimentPortal, self).__init__(workspace=experiment.workspace)
        if experiment._id:
            experiment_path = HasExperimentPortal.EXPERIMENT_ID_PATH.format(experiment._id)
        else:
            experiment_path = HasExperimentPortal.EXPERIMENT_NAME_PATH.format(experiment.name)
        self._experiment_url = \
            self.portal_url + \
            experiment_path + \
            HasWorkspacePortal.WORKSPACE_FMT.format(
                experiment.workspace.subscription_id,
                experiment.workspace.resource_group,
                experiment.workspace.name) + \
            self._formatted_tid

    def get_portal_url(self):
        """
        :return: Returns the Azure portal url for the experiment.
        :rtype: str
        """
        # type () -> str
        _warn_on_preview_url(self._experiment_url)
        return self._experiment_url


class HasRunPortal(HasExperimentPortal):
    """
    Mixin for providing run links to the Azure portal.
    :param experiment:
    :type experiment: Experiment
    :param run_id:
    :type run_id: str
    """

    RUN_PATH = '/runs/{0}'

    def __init__(self, experiment, run_id):
        """ The constructor method """
        super(HasRunPortal, self).__init__(experiment=experiment)
        self._run_details_url = \
            self.portal_url + \
            HasRunPortal.RUN_PATH.format(run_id) + \
            HasWorkspacePortal.WORKSPACE_FMT.format(
                experiment.workspace.subscription_id,
                experiment.workspace.resource_group,
                experiment.workspace.name) + \
            self._formatted_tid

    def get_portal_url(self):
        """
        :return: Returns the Azure portal url for the experiment.
        :rtype: str
        """
        # type () -> str
        _warn_on_preview_url(self._run_details_url)
        return self._run_details_url


class HasPipelinePortal(HasWorkspacePortal):
    """
    Mixin for providing pipeline links to the Azure portal.
    :param pipeline:
    :type pipeline: Pipeline
    """

    PIPELINE_PATH = '/pipelines/{0}'

    def __init__(self, pipeline):
        """ The constructor method """
        super(HasPipelinePortal, self).__init__(workspace=pipeline.workspace)
        self._pipeline_url =\
            self.portal_url + \
            HasPipelinePortal.PIPELINE_PATH.format(pipeline.id) + \
            HasWorkspacePortal.WORKSPACE_FMT.format(
                pipeline.workspace.subscription_id,
                pipeline.workspace.resource_group,
                pipeline.workspace.name)

    def get_portal_url(self):
        """
        :return: Returns the Azure portal url for the pipeline.
        :rtype: str
        """
        # type () -> str
        _warn_on_preview_url(self._pipeline_url)
        return self._pipeline_url
