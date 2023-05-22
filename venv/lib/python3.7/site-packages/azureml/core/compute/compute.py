# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the abstract parent and configuration classes for compute targets in Azure Machine Learning."""

try:
    from abc import ABCMeta
    ABC = ABCMeta('ABC', (), {})
except ImportError:
    from abc import ABC
from abc import abstractmethod
import json
import uuid
import requests
import sys
import time
import logging
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT, MLC_LIST_COMPUTES_FMT
from azureml._compute._constants import RP_COMPUTE_RESOURCE_ID_FMT, RP_LIST_COMPUTES_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml._base_sdk_common.user_agent import get_user_agent
from azureml._base_sdk_common import _ClientSessionId
from azureml._compute._util import get_paginated_compute_results
from azureml._compute._util import get_requests_session
from azureml.exceptions import ComputeTargetException, UserErrorException
from azureml._restclient.clientbase import ClientBase
from azureml._restclient.constants import RequestHeaders
from dateutil.parser import parse


class ComputeTarget(ABC):
    """Abstract parent class for all compute targets managed by Azure Machine Learning.

    A compute target is a designated compute resource/environment where you run your training script
    or host your service deployment. This location may be your local machine or a cloud-based compute resource.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`_

    .. remarks::

        Use the ComputeTarget constructor to retrieve
        the cloud representation of a Compute object associated with the provided workspace. The constructor
        returns an instance of a child class corresponding to the specific type of the retrieved Compute object.
        If the Compute object is not found, a :class:`azureml.exceptions.ComputeTargetException` is raised.

    :param workspace: The workspace object containing the Compute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the Compute object to retrieve.
    :type name: str
    """

    _compute_type = None

    def __new__(cls, workspace, name):
        """Return an instance of a compute target.

        ComputeTarget constructor is used to retrieve a cloud representation of a Compute object associated with the
        provided workspace. Will return an instance of a child class corresponding to the specific type of the
        retrieved Compute object.

        :param workspace: The workspace object containing the Compute object to retrieve.
        :type workspace: azureml.core.Workspace
        :param name: The name of the of the Compute object to retrieve.
        :type name: str
        :return: An instance of a child of :class:`azureml.core.ComputeTarget` corresponding to the
            specific type of the retrieved Compute object
        :rtype: azureml.core.ComputeTarget
        :raises azureml.exceptions.ComputeTargetException:
        """
        if workspace and name:
            compute_payload = cls._get(workspace, name)
            if compute_payload:
                compute_type = compute_payload['properties']['computeType']
                is_attached = compute_payload['properties']['isAttachedCompute']
                for child in ComputeTarget.__subclasses__():
                    if is_attached and compute_type == 'VirtualMachine' and child.__name__ == 'DsvmCompute':
                        # Cannot attach DsvmCompute
                        continue
                    elif not is_attached and compute_type == 'VirtualMachine' and child.__name__ == 'RemoteCompute':
                        # Cannot create RemoteCompute
                        continue
                    elif not is_attached and compute_type == 'Kubernetes' and child.__name__ == 'KubernetesCompute':
                        # Cannot create KubernetesCompute
                        continue
                    elif compute_type == child._compute_type:
                        compute_target = super(ComputeTarget, cls).__new__(child)
                        compute_target._initialize(workspace, compute_payload)
                        return compute_target
            else:
                raise ComputeTargetException('ComputeTargetNotFound: Compute Target with name {} not found in '
                                             'provided workspace'.format(name))
        else:
            return super(ComputeTarget, cls).__new__(cls)

    def __init__(self, workspace, name):
        """Class ComputeTarget constructor.

        Retrieve a cloud representation of a Compute object associated with the provided workspace. Returns an
        instance of a child class corresponding to the specific type of the retrieved Compute object.

        :param workspace: The workspace object containing the Compute object to retrieve.
        :type workspace: azureml.core.Workspace
        :param name: The name of the of the Compute object to retrieve.
        :type name: str
        :return: An instance of a child of :class:`azureml.core.ComputeTarget` corresponding to
            the specific type of the retrieved Compute object
        :rtype: azureml.core.ComputeTarget
        :raises azureml.exceptions.ComputeTargetException:
        """
        pass

    def __repr__(self):
        """Return the string representation of the ComputeTarget object.

        :return: String representation of the ComputeTarget object.
        :rtype: str
        """
        return "{}(workspace={}, name={}, id={}, type={}, provisioning_state={}, location={}, " \
               "tags={})".format(self.__class__.__name__,
                                 self.workspace.__repr__() if hasattr(self, 'workspace') else None,
                                 self.name if hasattr(self, 'name') else None,
                                 self.id if hasattr(self, 'id') else None,
                                 self.type if hasattr(self, 'type') else None,
                                 self.provisioning_state if hasattr(self, 'provisioning_state') else None,
                                 self.location if hasattr(self, 'location') else None,
                                 self.tags if hasattr(self, 'tags') else None,)

    @abstractmethod
    def _initialize(self, compute_resource_id, name, location, compute_type, tags, description, created_on,
                    modified_on, provisioning_state, provisioning_errors, cluster_resource_id, cluster_location,
                    workspace, mlc_endpoint, operation_endpoint, auth, is_attached):
        """Initilize abstract method.

        :param compute_resource_id:
        :type compute_resource_id: str
        :param name:
        :type name: str
        :param location:
        :type location: str
        :param compute_type:
        :type compute_type: str
        :param tags:
        :type tags: builtin.list[str]
        :param description:
        :type description: str
        :param created_on:
        :type created_on: datetime.datetime
        :param modified_on:
        :type modified_on: datetime.datetime
        :param provisioning_state:
        :type provisioning_state: str
        :param provisioning_errors:
        :type provisioning_errors: builtin.list[dict]
        :param cluster_resource_id:
        :type cluster_resource_id: str
        :param cluster_location:
        :type cluster_location: str
        :param workspace:
        :type workspace: azureml.core.Workspace
        :param mlc_endpoint:
        :type mlc_endpoint: str
        :param operation_endpoint:
        :type operation_endpoint: str
        :param auth:
        :type auth: azureml.core.authentication.AbstractAuthentication
        :param is_attached:
        :type is_attached: boolean
        :return:
        :rtype: None
        """
        self.id = compute_resource_id
        self.name = name
        self.location = location
        self.type = compute_type
        self.tags = tags
        self.description = description
        self.created_on = parse(created_on) if created_on else None
        self.modified_on = parse(modified_on) if modified_on else None
        self.provisioning_state = provisioning_state
        self.provisioning_errors = provisioning_errors
        self.cluster_resource_id = cluster_resource_id
        self.cluster_location = cluster_location
        self.workspace = workspace
        self._mlc_endpoint = mlc_endpoint
        self._operation_endpoint = operation_endpoint
        self._auth = auth
        self.is_attached = is_attached

    @staticmethod
    def _get_resource_manager_endpoint(workspace):
        """Return endpoint for resource manager based on cloud type.

        For AzureCloud, resource manager endpoint is: "https://management.azure.com/".

        :param workspace:
        :type workspace: azureml.core.Workspace
        :return:
        :rtype: str
        """
        return workspace._auth._get_cloud_type().endpoints.resource_manager

    @staticmethod
    def _get_compute_endpoint(workspace, name):
        """Return mlc endpoint for the compute.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :return:
        :rtype: str
        """
        compute_resource_id = MLC_COMPUTE_RESOURCE_ID_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                                 workspace.name, name)
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace)
        return '{}{}'.format(resource_manager_endpoint, compute_resource_id)

    @staticmethod
    def _get_list_computes_endpoint(workspace):
        """Return mlc endpoint for list computes.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :return:
        :rtype: str
        """
        list_computes = MLC_LIST_COMPUTES_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                     workspace.name)
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace)
        return '{}{}'.format(resource_manager_endpoint, list_computes)

    @staticmethod
    def _get_rp_compute_endpoint(workspace, name):
        """Return rp endpoint for the compute.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :return:
        :rtype: str
        """
        compute_resource_id = RP_COMPUTE_RESOURCE_ID_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                                 workspace.name, name)
        api_endpoint = workspace.service_context._get_api_url()
        return '{}{}'.format(api_endpoint, compute_resource_id)

    @staticmethod
    def _get_rp_list_computes_endpoint(workspace):
        """Return rp endpoint for list computes.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :return:
        :rtype: str
        """
        list_computes = RP_LIST_COMPUTES_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                     workspace.name)
        api_endpoint = workspace.service_context._get_api_url()
        return '{}{}'.format(api_endpoint, list_computes)

    @staticmethod
    def _get(workspace, name):
        """Return web response content for the compute.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :return:
        :rtype: dict
        """
        endpoint = ComputeTarget._get_rp_compute_endpoint(workspace, name)
        headers = workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().get, endpoint, params=params, headers=headers)
        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            get_content = json.loads(content)
            return get_content
        elif resp.status_code == 404:
            return None
        else:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))

    @staticmethod
    def create(workspace, name, provisioning_configuration):
        """Provision a Compute object by specifying a compute type and related configuration.

        This method creates a new compute target rather than attaching an existing one.

        .. remarks::

            The type of object provisioned is determined by the provisioning configuration provided.

            In the following example, a persistent compute target provisioned by
            :class:`azureml.core.compute.AmlCompute` is created. The ``provisioning_configuration`` parameter in this
            example is of type :class:`azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration`.

            .. code-block:: python

                from azureml.core.compute import ComputeTarget, AmlCompute
                from azureml.core.compute_target import ComputeTargetException

                # Choose a name for your CPU cluster
                cpu_cluster_name = "cpu-cluster"

                # Verify that cluster does not exist already
                try:
                    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
                    print('Found existing cluster, use it.')
                except ComputeTargetException:
                    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                                           max_nodes=4)
                    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

                cpu_cluster.wait_for_completion(show_output=True)

            Full sample is available from
            https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb


        :param workspace: The workspace object to create the Compute object under.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the Compute object.
        :type name: str
        :param provisioning_configuration: A ComputeTargetProvisioningConfiguration object that is used to determine
            the type of Compute object to provision, and how to configure it.
        :type provisioning_configuration: azureml.core.compute.compute.ComputeTargetProvisioningConfiguration
        :return: An instance of a child of ComputeTarget corresponding to the type of object provisioned.
        :rtype: azureml.core.ComputeTarget
        :raises azureml.exceptions.ComputeTargetException:
        """
        if name in ["amlcompute", "local", "containerinstance"]:
            raise UserErrorException("Please specify a different target name."
                                     " {} is a reserved name.".format(name))
        compute_type = provisioning_configuration._compute_type
        return compute_type._create(workspace, name, provisioning_configuration)

    @staticmethod
    def _create_compute_target(workspace, name, compute_payload, target_class):
        """Create compute target.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param compute_payload:
        :type compute_payload: dict
        :param target_class:
        :type target_class:
        :return:
        :rtype: azureml.core.ComputeTarget
        """
        compute_location = compute_payload.get('properties').get('computeLocation')
        if compute_location and compute_location != workspace.location:
            logging.warning("You might see network latency and increased data transfer costs if you chose a cluster "
                            "location different from the location of your workspace")
        endpoint = ComputeTarget._get_compute_endpoint(workspace, name)
        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth.get_authentication_header())
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().put, endpoint, params=params, headers=headers,
                                        json=compute_payload)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        if 'Azure-AsyncOperation' not in resp.headers:
            raise ComputeTargetException('Error, missing operation location from resp headers:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        compute_target = target_class(workspace, name)
        compute_target._operation_endpoint = resp.headers['Azure-AsyncOperation']
        return compute_target

    @staticmethod
    def attach(workspace, name, attach_configuration):
        """Attach a Compute object to a workspace using the specified name and configuration information.

        .. remarks::

            The type of object to pass to the parameter ``attach_configuration`` is a
            :class:`azureml.core.compute.compute.ComputeTargetAttachConfiguration`
            object built using the ``attach_configuration`` function on any of the child classes of
            :class:`azureml.core.ComputeTarget`.

            The following example shows how to attach an ADLA account to a workspace using the
            :meth:`azureml.core.compute.AdlaCompute.attach_configuration` method of AdlaCompute.

            .. code-block:: python

                adla_compute_name = 'testadl' # Name to associate with new compute in workspace

                # ADLA account details needed to attach as compute to workspace
                adla_account_name = "<adla_account_name>" # Name of the Azure Data Lake Analytics account
                adla_resource_group = "<adla_resource_group>" # Name of the resource group which contains this account

                try:
                    # check if already attached
                    adla_compute = AdlaCompute(ws, adla_compute_name)
                except ComputeTargetException:
                    print('attaching adla compute...')
                    attach_config = AdlaCompute.attach_configuration(resource_group=adla_resource_group, account_name=adla_account_name)
                    adla_compute = ComputeTarget.attach(ws, adla_compute_name, attach_config)
                    adla_compute.wait_for_completion()

                print("Using ADLA compute:{}".format(adla_compute.cluster_resource_id))
                print("Provisioning state:{}".format(adla_compute.provisioning_state))
                print("Provisioning errors:{}".format(adla_compute.provisioning_errors))

            Full sample is available from
            https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-use-adla-as-compute-target.ipynb


        :param workspace: The workspace object to attach the Compute object to.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the Compute object.
        :type name: str
        :param attach_configuration: A ComputeTargetAttachConfiguration object that is used to determine
            the type of Compute object to attach, and how to configure it.
        :type attach_configuration: azureml.core.compute.compute.ComputeTargetAttachConfiguration
        :return: An instance of a child of ComputeTarget corresponding to the type of object attached.
        :rtype: azureml.core.ComputeTarget
        :raises azureml.exceptions.ComputeTargetException:
        """
        compute_type = attach_configuration._compute_type
        return compute_type._attach(workspace, name, attach_configuration)

    @staticmethod
    def _attach(workspace, name, attach_payload, target_class):
        """Attach implementation method.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param attach_payload:
        :type attach_payload: dict
        :param target_class:
        :type target_class:
        :return:
        :rtype:
        """
        attach_payload['location'] = workspace.location
        endpoint = ComputeTarget._get_compute_endpoint(workspace, name)
        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth.get_authentication_header())
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().put, endpoint, params=params, headers=headers,
                                        json=attach_payload)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        if 'Azure-AsyncOperation' not in resp.headers:
            raise ComputeTargetException('Error, missing operation location from resp headers:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        compute_target = target_class(workspace, name)
        compute_target._operation_endpoint = resp.headers['Azure-AsyncOperation']
        return compute_target

    @staticmethod
    def list(workspace):
        """List all ComputeTarget objects within the workspace.

        Return a list of instantiated child objects corresponding to the specific type of Compute. Objects are
        children of :class:`azureml.core.ComputeTarget`.

        :param workspace: The workspace object containing the objects to list.
        :type workspace: azureml.core.Workspace
        :return: List of compute targets within the workspace.
        :rtype: builtin.list[azureml.core.ComputeTarget]
        :raises azureml.exceptions.ComputeTargetException:
        """
        envs = []
        endpoint = ComputeTarget._get_rp_list_computes_endpoint(workspace)
        headers = workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().get, endpoint, params=params, headers=headers)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Error occurred retrieving targets:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        is_windows_contrib_installed = True
        try:
            from azureml.contrib.compute import AmlWindowsCompute  # noqa: F401
        except ImportError:
            is_windows_contrib_installed = False
            pass
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        result_list = json.loads(content)
        paginated_results = get_paginated_compute_results(result_list, headers)
        for env in paginated_results:
            if 'properties' in env and 'computeType' in env['properties']:
                compute_type = env['properties']['computeType']
                is_attached = env['properties']['isAttachedCompute']
                env_obj = None
                for child in ComputeTarget.__subclasses__():
                    if is_attached and compute_type == 'VirtualMachine' and child.__name__ == 'DsvmCompute':
                        # Cannot attach DsvmCompute
                        continue
                    elif not is_attached and compute_type == 'VirtualMachine' and child.__name__ == 'RemoteCompute':
                        # Cannot create RemoteCompute
                        continue
                    elif not is_attached and compute_type == 'Kubernetes' and child.__name__ == 'KubernetesCompute':
                        # Cannot create KubernetesCompute
                        continue
                    elif compute_type == child._compute_type:
                        # If windows contrib is not installed, don't list windows compute type
                        # Windows is currently supported only for RL runs.
                        # The windows contrib is installed as a part of RL SDK install.
                        # This step is trying to avoid users using this compute target by mistake for a non-RL run
                        if not is_windows_contrib_installed and "properties" in env['properties'] and \
                                env['properties']['properties'] is not None and \
                                "osType" in env['properties']['properties'] and \
                                env['properties']['properties']['osType'].lower() == 'windows':
                            pass
                        else:
                            env_obj = child.deserialize(workspace, env)
                        break
                if env_obj:
                    envs.append(env_obj)
        return envs

    def wait_for_completion(self, show_output=False, is_delete_operation=False):
        """Wait for the current provisioning operation to finish on the cluster.

        This method returns a :class:`azureml.exceptions.ComputeTargetException` if there is a problem
        polling the compute object.

        :param show_output: Indicates whether to provide more verbose output.
        :type show_output: bool
        :param is_delete_operation: Indicates whether the operation is meant for deleting.
        :type is_delete_operation: bool
        :raises azureml.exceptions.ComputeTargetException:
        """
        try:
            operation_state, error = self._wait_for_completion(show_output)
            print('Provisioning operation finished, operation "{}"'.format(operation_state))
            if not is_delete_operation:
                self.refresh_state()
            if operation_state != 'Succeeded':
                if error and 'statusCode' in error and 'message' in error:
                    error_response = ('StatusCode: {}\n'
                                      'Message: {}'.format(error['statusCode'], error['message']))
                else:
                    error_response = error

                raise ComputeTargetException('Compute object provisioning polling reached non-successful terminal '
                                             'state, current provisioning state: {}\n'
                                             'Provisioning operation error:\n'
                                             '{}'.format(self.provisioning_state, error_response))
        except ComputeTargetException as e:
            if e.message == 'No operation endpoint':
                self.refresh_state()
                raise ComputeTargetException('Long running operation information not known, unable to poll. '
                                             'Current state is {}'.format(self.provisioning_state))
            else:
                raise e

    def _wait_for_completion(self, show_output):
        """Wait for completion implementation.

        :param show_output:
        :type show_output: bool
        :return:
        :rtype: (str, dict)
        """
        if not self._operation_endpoint:
            raise ComputeTargetException('No operation endpoint')
        operation_state, error = self._get_operation_state()
        current_state = operation_state
        if show_output:
            sys.stdout.write('{}'.format(current_state))
            sys.stdout.flush()
        while operation_state != 'Succeeded' and operation_state != 'Failed' and operation_state != 'Canceled':
            time.sleep(5)
            operation_state, error = self._get_operation_state()
            if show_output:
                sys.stdout.write('.')
                if operation_state != current_state:
                    sys.stdout.write('\n{}'.format(operation_state))
                    current_state = operation_state
                sys.stdout.flush()
        return operation_state, error

    def _get_operation_state(self):
        """Return operation state.

        :return:
        :rtype: (str, dict)
        """
        headers = self._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {}

        # API version should not be appended for operation status URLs.
        # This is a bug fix for older SDK and ARM breaking changes and
        # will append version only if the request URL doesn't have one.
        if 'api-version' not in self._operation_endpoint:
            params = {'api-version': MLC_WORKSPACE_API_VERSION}

        resp = ClientBase._execute_func(get_requests_session().get, self._operation_endpoint, params=params,
                                        headers=headers)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = json.loads(content)
        status = content['status']
        error = content.get('error')

        # Prior to API version 2019-06-01 the 'error' element was double nested.
        # This change retains backwards compat for 2018-11-19 version.
        if error is not None:
            innererror = error.get('error')
            if innererror is not None:
                error = innererror
        # ---------------------------------------------------------------------

        return status, error

    @staticmethod
    def _add_request_tracking_headers(headers):
        if RequestHeaders.CLIENT_REQUEST_ID not in headers:
            headers[RequestHeaders.CLIENT_REQUEST_ID] = str(uuid.uuid4())

        if RequestHeaders.CLIENT_SESSION_ID not in headers:
            headers[RequestHeaders.CLIENT_SESSION_ID] = _ClientSessionId

        if RequestHeaders.USER_AGENT not in headers:
            headers[RequestHeaders.USER_AGENT] = get_user_agent()

    @abstractmethod
    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        Update properties based on the current state of the corresponding cloud object.
        This is useful for manual polling of compute state.

        This abstract method is implemented by child classes of :class:`azureml.core.ComputeTarget`.
        """
        pass

    def get_status(self):
        """Retrieve the current provisioning state of the Compute object.

        .. remarks::

            Values returned are listed in the Azure REST API Reference for
            `ProvisioningState <https://docs.microsoft.com/rest/api/azureml/workspacesandcomputes
            /machinelearningcompute/get#provisioningstate>`_.

        :return: The current ``provisioning_state``.
        :rtype: str
        """
        self.refresh_state()
        return self.provisioning_state

    @abstractmethod
    def delete(self):
        """Remove the Compute object from its associated workspace.

        This abstract method is implemented by child classes of :class:`azureml.core.ComputeTarget`.

        .. remarks::

            If this object was created through Azure Machine Learning, the corresponding cloud-based objects
            will also be deleted. If this object was created externally and only attached to the workspace, this
            method raises an exception and nothing is changed.
        """
        pass

    @abstractmethod
    def detach(self):
        """Detach the Compute object from its associated workspace.

        This abstract method is implemented by child classes of :class:`azureml.core.ComputeTarget`.
        Underlying cloud objects are not deleted, only their associations are removed.
        """
        pass

    def _delete_or_detach(self, underlying_resource_action):
        """Remove the Compute object from its associated workspace.

        If underlying_resource_action is 'delete', the corresponding cloud-based objects will also be deleted.
        If underlying_resource_action is 'detach', no underlying cloud object will be deleted, the association
        will just be removed.

        :param underlying_resource_action: whether delete or detach the underlying cloud object
        :type underlying_resource_action: str
        :raises azureml.exceptions.ComputeTargetException:
        """
        headers = self._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION, 'underlyingResourceAction': underlying_resource_action}
        resp = ClientBase._execute_func(get_requests_session().delete, self._mlc_endpoint, params=params,
                                        headers=headers)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))

        self.provisioning_state = 'Deleting'
        self._operation_endpoint = resp.headers['Azure-AsyncOperation']

    @abstractmethod
    def serialize(self):
        """Convert this Compute object into a JSON serialized dictionary.

        :return: The JSON representation of this Compute object.
        :rtype: dict
        """
        created_on = self.created_on.isoformat() if self.created_on else None
        modified_on = self.modified_on.isoformat() if self.modified_on else None
        compute = {'id': self.id, 'name': self.name, 'location': self.location, 'type': self.type, 'tags': self.tags,
                   'description': self.description, 'created_on': created_on, 'modified_on': modified_on,
                   'provisioning_state': self.provisioning_state, 'provisioning_errors': self.provisioning_errors}
        return compute

    @staticmethod
    @abstractmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into a Compute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the Compute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a Compute object.
        :type object_dict: dict
        :return: The Compute representation of the provided JSON object.
        :rtype: azureml.core.ComputeTarget
        """
        pass

    @staticmethod
    @abstractmethod
    def _validate_get_payload(payload):
        pass


class ComputeTargetProvisioningConfiguration(ABC):
    """Abstract parent class for all ComputeTarget provisioning configuration objects.

    This class defines the configuration parameters for provisioning
    compute objects. Provisioning configuration varies by child compute object. Specify provisioning configuration
    with the ``provisioning_configuration`` method of child compute objects that require provisioning.

    :param type: The type of ComputeTarget this object is associated with.
    :type type: azureml.core.ComputeTarget
    :param location: The Azure region to provision the Compute object in.
    :type location: str
    """

    def __init__(self, type, location):
        """Initialize the ProvisioningConfiguration object.

        :param type: The type of ComputeTarget this object is associated with
        :type type: azureml.core.ComputeTarget
        :param location: The Azure region to provision the Compute object in.
        :type location: str
        :return: The ProvisioningConfiguration object
        :rtype: azureml.core.compute.compute.ComputeTargetProvisioningConfiguration
        """
        self._compute_type = type
        self.location = location

    @abstractmethod
    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        pass


class ComputeTargetAttachConfiguration(ABC):
    """Abstract parent class for all ComputeTarget attach configuration objects.

    This class defines the configuration parameters for attaching compute objects.
    Attach configuration varies by child compute object. Specify attach configuration with the
    ``attach_configuration`` method of child compute objects.

    :param type: The type of ComputeTarget this object is associated with.
    :type type: azureml.core.ComputeTarget
    """

    def __init__(self, type):
        """Initialize the AttachConfiguration object.

        :param type: The type of ComputeTarget this object is associated with.
        :type type: azureml.core.ComputeTarget
        :return: The AttachConfiguration object.
        :rtype: azureml.core.compute.compute.ComputeTargetAttachConfiguration
        """
        self._compute_type = type

    @abstractmethod
    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        pass


class ComputeTargetUpdateConfiguration(ABC):
    """Abstract parent class for all ComputeTarget update configuration objects.

    This class defines configuration parameters for updating compute objects.
    Update configuration varies by child compute object. Specify update configuration
    with the ``update`` method of child compute objects that support updating.
    """

    def __init__(self, type):
        """Initialize the UpdateConfiguration object.

        :param compute: The type of ComputeTarget that should be updated.
        :type compute: azureml.core.ComputeTarget
        :return: The ComputeTargetUpdateConfiguration object.
        :rtype: azureml.core.compute.compute.ComputeTargetUpdateConfiguration
        """
        self._compute_type = type

    @abstractmethod
    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        pass
