# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Azure Machine Learning compute targets in Azure Machine Learning."""

import copy
import json
import requests
import string
import sys
import time
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml._compute._util import amlcompute_payload_template
from azureml._compute._util import convert_duration_to_seconds
from azureml._compute._util import convert_seconds_to_duration
from azureml._compute._util import get_paginated_compute_nodes
from azureml._compute._util import get_paginated_compute_results
from azureml._compute._util import get_paginated_compute_supported_vms
from azureml._compute._util import get_requests_session
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetProvisioningConfiguration
from azureml.exceptions import ComputeTargetException
from azureml._restclient.clientbase import ClientBase
from azureml._restclient.workspace_client import WorkspaceClient
from azureml._restclient.constants import DEFAULT_PAGE_SIZE
from dateutil.parser import parse


class AmlCompute(ComputeTarget):
    """Manages an Azure Machine Learning compute in Azure Machine Learning.

    An Azure Machine Learning Compute (AmlCompute) is a managed-compute infrastructure that
    allows you to easily create a single or multi-node compute. The compute is created within your workspace
    region as a resource that can be shared with other users.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`_

    .. remarks::

        In the following example, a persistent compute target provisioned by
        :class:`azureml.core.compute.AmlCompute` is created. The ``provisioning_configuration`` parameter in this
        example is of type :class:`azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration`, which is
        a child class of :class:`azureml.core.compute.compute.ComputeTargetProvisioningConfiguration`.

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


    :param workspace: The workspace object containing the AmlCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the AmlCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'AmlCompute'

    def _initialize(self, workspace, obj_dict):
        """Initialize implementation method.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        name = obj_dict['name']
        compute_resource_id = MLC_COMPUTE_RESOURCE_ID_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                                 workspace.name, name)
        resource_manager_endpoint = self._get_resource_manager_endpoint(workspace).rstrip('/')
        mlc_endpoint = '{}{}'.format(resource_manager_endpoint, compute_resource_id)
        location = obj_dict['location']
        compute_type = obj_dict['properties']['computeType']
        tags = obj_dict['tags']
        description = obj_dict['properties']['description']
        created_on = obj_dict['properties'].get('createdOn')
        modified_on = obj_dict['properties'].get('modifiedOn')
        cluster_resource_id = obj_dict['properties']['resourceId']
        cluster_location = obj_dict['properties']['computeLocation'] \
            if 'computeLocation' in obj_dict['properties'] else None
        provisioning_state = obj_dict['properties']['provisioningState']
        provisioning_errors = obj_dict['properties']['provisioningErrors']
        is_attached = obj_dict['properties']['isAttachedCompute'] \
            if 'isAttachedCompute' in obj_dict['properties'] else None
        vm_size = obj_dict['properties']['properties']['vmSize'] \
            if obj_dict['properties']['properties'] else None
        vm_priority = obj_dict['properties']['properties']['vmPriority'] \
            if obj_dict['properties']['properties'] else None
        scale_settings = obj_dict['properties']['properties']['scaleSettings'] \
            if obj_dict['properties']['properties'] else None
        scale_settings = ScaleSettings.deserialize(scale_settings) if scale_settings else None
        admin_username = obj_dict['properties']['properties']['userAccountCredentials']['adminUserName'] \
            if obj_dict['properties']['properties'] and \
            'userAccountCredentials' in obj_dict['properties']['properties'] and \
            'adminUserName' in obj_dict['properties']['properties']['userAccountCredentials'] else None
        admin_user_password = obj_dict['properties']['properties']['userAccountCredentials']['adminUserPassword'] \
            if obj_dict['properties']['properties'] and \
            'userAccountCredentials' in obj_dict['properties']['properties'] and \
            'adminUserPassword' in obj_dict['properties']['properties']['userAccountCredentials'] else None
        admin_user_ssh_key = obj_dict['properties']['properties']['userAccountCredentials']['adminUserSshPublicKey'] \
            if obj_dict['properties']['properties'] and \
            'userAccountCredentials' in obj_dict['properties']['properties'] and \
            'adminUserSshPublicKey' in obj_dict['properties']['properties']['userAccountCredentials'] else None
        vnet_resourcegroup_name = None
        vnet_name = None
        subnet_name = None
        subnet_id = obj_dict['properties']['properties']['subnet']['id'] \
            if obj_dict['properties']['properties'] and obj_dict['properties']['properties']['subnet'] and \
            'id' in obj_dict['properties']['properties']['subnet'] else None
        if subnet_id:
            vnet_resourcegroup_name = \
                subnet_id[subnet_id.index("/resourceGroups/") + len("/resourceGroups/"):subnet_id.index("/providers")]
            vnet_name = \
                subnet_id[subnet_id.index("/virtualNetworks/") + len("/virtualNetworks/"):subnet_id.index("/subnets")]
            subnet_name = subnet_id[subnet_id.index("/subnets/") + len("/subnets/"):]
        remote_login_port_public_access = obj_dict['properties']['properties']['remoteLoginPortPublicAccess'] \
            if obj_dict['properties']['properties'] and \
            'remoteLoginPortPublicAccess' in obj_dict['properties']['properties'] else None
        status = AmlComputeStatus.deserialize(obj_dict['properties']) \
            if provisioning_state in ["Succeeded", "Updating"] else None
        identity_type = obj_dict['identity']['type'] \
            if 'identity' in obj_dict else None
        identity_id = obj_dict['identity']['userAssignedIdentities'] \
            if identity_type and 'userAssignedIdentities' in obj_dict['identity'] else None
        super(AmlCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags, description,
                                            created_on, modified_on, provisioning_state, provisioning_errors,
                                            cluster_resource_id, cluster_location, workspace, mlc_endpoint, None,
                                            workspace._auth, is_attached)
        self.vm_size = vm_size
        self.vm_priority = vm_priority
        self.scale_settings = scale_settings
        self.admin_username = admin_username
        self.admin_user_password = admin_user_password
        self.admin_user_ssh_key = admin_user_ssh_key
        self.vnet_resourcegroup_name = vnet_resourcegroup_name
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name
        self.remote_login_port_public_access = remote_login_port_public_access
        self.status = status
        self.identity_type = identity_type
        self.identity_id = identity_id

    def __repr__(self):
        """Return the string representation of the AmlCompute object.

        :return: String representation of the AmlCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def _create(workspace, name, provisioning_configuration):
        """Create implementation method.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param provisioning_configuration:
        :type provisioning_configuration: azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration
        :return:
        :rtype: azureml.core.compute.amlcompute.AmlCompute
        """
        compute_create_payload = AmlCompute._build_create_payload(provisioning_configuration,
                                                                  workspace.location, workspace.subscription_id)
        return ComputeTarget._create_compute_target(workspace, name, compute_create_payload, AmlCompute)

    @staticmethod
    def provisioning_configuration(vm_size='', vm_priority="dedicated", min_nodes=0,
                                   max_nodes=None, idle_seconds_before_scaledown=1800,
                                   admin_username=None, admin_user_password=None, admin_user_ssh_key=None,
                                   vnet_resourcegroup_name=None, vnet_name=None, subnet_name=None,
                                   tags=None, description=None, remote_login_port_public_access="NotSpecified",
                                   identity_type=None, identity_id=None, location=None, enable_node_public_ip=True):
        """Create a configuration object for provisioning an AmlCompute target.

        :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
            Note that not all sizes are available in all regions, as
            detailed in the previous link. If not specified, defaults to Standard_NC6.
        :type vm_size: str
        :param vm_priority: The VM priority, `dedicated` or `lowpriority`.
        :type vm_priority: str
        :param min_nodes: The minimum number of nodes to use on the cluster. If not specified, defaults to 0.
        :type min_nodes: int
        :param max_nodes: The maximum number of nodes to use on the cluster. If not specified, defaults to 4.
        :type max_nodes: int
        :param idle_seconds_before_scaledown: Node idle time in seconds before scaling down the cluster.
            If not specified, defaults to 1800.
        :type idle_seconds_before_scaledown: int
        :param admin_username: The name of the administrator user account which can be used to SSH into nodes.
        :type admin_username: str
        :param admin_user_password: The password of the administrator user account.
        :type admin_user_password: str
        :param admin_user_ssh_key: The SSH public key of the administrator user account.
        :type admin_user_ssh_key: str
        :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
        :type vnet_resourcegroup_name: str
        :param vnet_name: The name of the virtual network.
        :type vnet_name: str
        :param subnet_name: The name of the subnet inside the VNet.
        :type subnet_name: str
        :param tags: A dictionary of key value tags to provide to the compute object.
        :type tags: dict[str, str]
        :param description: A description to provide to the compute object.
        :type description: str
        :param remote_login_port_public_access: State of the public SSH port. Possible values are:

            * Disabled - Indicates that the public ssh port is closed on all nodes of the cluster.
            * Enabled - Indicates that the public ssh port is open on all nodes of the cluster.
            * NotSpecified - Indicates that the public ssh port is closed on all nodes of the cluster if
              VNet is defined, else is open all public nodes. It can be this default value only during
              cluster creation time. After creation, it will be either enabled or disabled.

        :type remote_login_port_public_access: str
        :param identity_type: Possible values are:

            * SystemAssigned - System assigned identity
            * UserAssigned - User assigned identity. Requires identity id to be set.

        :type identity_type: string
        :param identity_id: List of resource ids for the user assigned identity. eg.
            ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<id>']
        :type identity_id: builtin.list[str]
        :param location: Location to provision cluster in.
        :type location: str
        :param enable_node_public_ip: Enable node public IP. Possible values are:

            * True - Enable node public IP.
            * False - Disable node public IP.
            * NotSpecified - Enable node public IP.
        :type enable_node_public_ip: bool
        :return: A configuration object to be used when creating a Compute object.
        :rtype: azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        config = AmlComputeProvisioningConfiguration(vm_size, vm_priority, min_nodes,
                                                     max_nodes, idle_seconds_before_scaledown,
                                                     admin_username, admin_user_password, admin_user_ssh_key,
                                                     vnet_resourcegroup_name, vnet_name, subnet_name,
                                                     tags, description, remote_login_port_public_access,
                                                     identity_type, identity_id, location, enable_node_public_ip)
        return config

    @staticmethod
    def _build_create_payload(config, location, subscription_id):
        """Construct the payload needed to create an AmlCompute cluster.

        :param config:
        :type config: azureml.core.compute.AmlComputeProvisioningConfiguration
        :param location:
        :type location: str
        :param subscription_id:
        :type subscription_id:
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(amlcompute_payload_template)
        del(json_payload['properties']['resourceId'])
        if not config.compute_location:
            del(json_payload['properties']['computeLocation'])
        else:
            json_payload['properties']['computeLocation'] = config.compute_location
        json_payload['location'] = location
        if not config.vm_size and not config.vm_priority and not config.admin_username and not \
                config.vnet_resourcegroup_name and not config.vnet_name and not config.subnet_name \
                and not config.remote_login_port_public_access:
            del(json_payload['properties']['properties'])
        else:
            if not config.vm_size:
                del(json_payload['properties']['properties']['vmSize'])
            else:
                json_payload['properties']['properties']['vmSize'] = config.vm_size
            if not config.vm_priority:
                del(json_payload['properties']['properties']['vmPriority'])
            else:
                json_payload['properties']['properties']['vmPriority'] = config.vm_priority
            json_payload['properties']['properties']['scaleSettings'] = config.scale_settings.serialize()
            if not config.admin_username:
                del(json_payload['properties']['properties']['userAccountCredentials'])
            else:
                json_payload['properties']['properties']['userAccountCredentials']['adminUserName'] = \
                    config.admin_username
                if config.admin_user_password:
                    json_payload['properties']['properties']['userAccountCredentials']['adminUserPassword'] = \
                        config.admin_user_password
                else:
                    del(json_payload['properties']['properties']['userAccountCredentials']['adminUserPassword'])
                if config.admin_user_ssh_key:
                    json_payload['properties']['properties']['userAccountCredentials']['adminUserSshPublicKey'] = \
                        config.admin_user_ssh_key
                else:
                    del(json_payload['properties']['properties']['userAccountCredentials']['adminUserSshPublicKey'])

            if not config.vnet_name:
                del(json_payload['properties']['properties']['subnet'])
                del(json_payload['properties']['properties']['enableNodePublicIp'])
            else:
                json_payload['properties']['properties']['subnet'] = \
                    {"id": "/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks"
                     "/{2}/subnets/{3}".format(subscription_id, config.vnet_resourcegroup_name,
                                               config.vnet_name, config.subnet_name)}
                json_payload['properties']['properties']['enableNodePublicIp'] = \
                    config.enable_node_public_ip
            if not config.remote_login_port_public_access:
                del(json_payload['properties']['properties']['remoteLoginPortPublicAccess'])
            else:
                json_payload['properties']['properties']['remoteLoginPortPublicAccess'] = \
                    config.remote_login_port_public_access

        if config.tags:
            json_payload['tags'] = config.tags
        else:
            del(json_payload['tags'])
        if config.description:
            json_payload['properties']['description'] = config.description
        else:
            del(json_payload['properties']['description'])
        if config.identity_type:
            json_payload['identity']['type'] = config.identity_type
            if config.identity_id:
                AmlCompute._build_identity_id_payload(json_payload, config.identity_id)
            else:
                del(json_payload['identity']['userAssignedIdentities'])
        else:
            del(json_payload['identity'])
        return json_payload

    @staticmethod
    def _build_identity_id_payload(json_payload, identity_id):
        """Construct the payload needed to create assign identity to AmlCompute cluster.

        :param json_payload:
        :type json_payload: dict
        :param identity_id: User assigned identities
        :type identity_id: builtin.list[str]
        """
        for id in identity_id:
            json_payload['identity']['userAssignedIdentities'][id] = dict()

    def wait_for_completion(self,
                            show_output=False,
                            min_node_count=None,
                            timeout_in_minutes=25,
                            is_delete_operation=False):
        """Wait for the AmlCompute cluster to finish provisioning.

        This can be configured to wait for a minimum number of nodes, and to timeout after a set period of time.

        :param show_output: Boolean to provide more verbose output.
        :type show_output: bool
        :param min_node_count: Minimum number of nodes to wait for before considering provisioning to be complete. This
            doesn't have to equal the minimum number of nodes that the compute was provisioned with, however it should
            not be greater than that.
        :type min_node_count: int
        :param timeout_in_minutes: The duration in minutes to wait before considering provisioning to have failed.
        :type timeout_in_minutes: int
        :param is_delete_operation: Indicates whether the operation is meant for deleting.
        :type is_delete_operation: bool
        :raises azureml.exceptions.ComputeTargetException:
        """
        # For all LROs poll MLC's operation status endpoint first
        if self._operation_endpoint:
            super(AmlCompute, self).wait_for_completion(show_output=show_output,
                                                        is_delete_operation=is_delete_operation)
        if not is_delete_operation:
            min_nodes_reached, timeout_reached, terminal_state_reached, status_errors_present = \
                self._wait_for_nodes(min_node_count, timeout_in_minutes, show_output)
            if show_output:
                print('AmlCompute wait for completion finished\n')
            if min_nodes_reached:
                if show_output:
                    print('Minimum number of nodes requested have been provisioned')
            elif timeout_reached:
                if show_output:
                    print('Wait timeout has been reached')
                    if self.status:
                        print('Current provisioning state of AmlCompute is "{0}" and current node count is "{1}"'.
                              format(self.status.provisioning_state.capitalize(), self.status.current_node_count))
                    else:
                        print('Current provisioning state of AmlCompute is "{}"'.
                              format(self.provisioning_state.capitalize()))
            elif terminal_state_reached:
                if self.status:
                    state = self.status.provisioning_state.capitalize()
                else:
                    state = self.provisioning_state.capitalize()
                if show_output:
                    print('Terminal state of "{}" has been reached\n'.format(state))
                    if state == 'Failed':
                        print('Provisioning errors: {}\n'.format(self.provisioning_errors))
            elif status_errors_present:
                if self.status:
                    errors = self.status.errors
                else:
                    errors = self.provisioning_errors
                if show_output:
                    print('There were errors reported from AmlCompute:\n{}'.format(errors))

    def _wait_for_nodes(self, min_node_count, timeout_in_minutes, show_output):
        """Wait for nodes.

        :param min_node_count:
        :type min_node_count: int
        :param timeout_in_minutes:
        :type timeout_in_minutes: int
        :param show_output:
        :type show_output: bool
        :return:
        :rtype:
        """
        self.refresh_state()
        start_time = time.time()

        if self.status:
            current_state = self.status.provisioning_state
        else:
            current_state = self.provisioning_state
        if show_output and current_state:
            sys.stdout.write('{}'.format(current_state))
            sys.stdout.flush()

        min_node_count = self._get_default_min_node_count(min_node_count)
        min_nodes_reached = self._min_node_count_reached(min_node_count)
        timeout_reached = self._polling_timeout_reached(start_time, timeout_in_minutes)
        terminal_state_reached = self._terminal_state_reached()
        status_errors_present = self._status_errors_present()

        while not min_nodes_reached and not timeout_reached and not terminal_state_reached \
                and not status_errors_present:
            if show_output and not min_nodes_reached:
                if self.status and self.status.current_node_count:
                    sys.stdout.write("Waiting for cluster to scale. {0} out of {1} nodes ready.\n"
                                     .format(self.status.current_node_count, min_node_count))

            time.sleep(5)
            self.refresh_state()

            if self.status:
                state = self.status.provisioning_state
            else:
                state = None

            if show_output and state:
                if state != current_state and state != 'succeeded':
                    if current_state is None:
                        sys.stdout.write('{}'.format(state))
                    else:
                        sys.stdout.write('\n{}'.format(state))
                    current_state = state
                elif state:
                    sys.stdout.write('.')
                sys.stdout.flush()

            min_nodes_reached = self._min_node_count_reached(min_node_count)
            timeout_reached = self._polling_timeout_reached(start_time, timeout_in_minutes)
            terminal_state_reached = self._terminal_state_reached()
            status_errors_present = self._status_errors_present()

        if show_output:
            sys.stdout.write('\n')
            sys.stdout.flush()

        return min_nodes_reached, timeout_reached, terminal_state_reached, status_errors_present

    def _get_default_min_node_count(self, min_node_count):
        """Return minimum node count.

        :param min_node_count:
        :type min_node_count: int
        :return:
        :rtype: int
        """
        if not min_node_count:
            if self.status and self.status.scale_settings:
                min_node_count = self.status.scale_settings.minimum_node_count
        return min_node_count

    def _min_node_count_reached(self, min_node_count):
        """Return minimum node count reached.

        :param min_node_count:
        :type min_node_count: int
        :return:
        :rtype: bool
        """
        min_node_count = self._get_default_min_node_count(min_node_count)
        if min_node_count is not None and self.status and self.status.current_node_count is not None and \
           self.status.current_node_count >= min_node_count:
            return True
        return False

    def _polling_timeout_reached(self, start_time, timeout_in_minutes):
        """Poll timeout reached.

        :param start_time:
        :type start_time: datetime.datetime
        :param timeout_in_minutes:
        :type timeout_in_minutes: int
        :return:
        :rtype: bool
        """
        cur_time = time.time()
        if cur_time - start_time > timeout_in_minutes * 60:
            return True
        return False

    def _terminal_state_reached(self):
        """Terminal state reached.

        :param state:
        :type state: str
        :return:
        :rtype: bool
        """
        if self.status:
            state = self.status.provisioning_state.capitalize()
        else:
            state = self.provisioning_state.capitalize()
        if state == 'Failed' or state == 'Canceled':
            return True
        return False

    def _status_errors_present(self):
        """Return status error.

        :return:
        :rtype:
        """
        if (self.status and self.status.errors) or self.provisioning_errors:
            return True
        return False

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = AmlCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location
        self.vm_size = cluster.vm_size
        self.vm_priority = cluster.vm_priority
        self.scale_settings = cluster.scale_settings
        self.status = cluster.status
        self.remote_login_port_public_access = cluster.remote_login_port_public_access
        self.identity_type = self.identity_type
        self.identity_id = self.identity_id

    def get_status(self):
        """Retrieve the current detailed status for the AmlCompute cluster.

        :return: A detailed status object for the cluster
        :rtype: azureml.core.compute.amlcompute.AmlComputeStatus
        """
        self.refresh_state()
        if not self.status:
            state = self.provisioning_state.capitalize()
            if state == 'Creating':
                print('AmlCompute is getting created. Consider calling wait_for_completion() first\n')
            elif state == 'Failed':
                print('AmlCompute is in a failed state, try deleting and recreating\n')
            else:
                print('Current provisioning state of AmlCompute is "{}"\n'.format(state))
            return None

        return self.status

    def delete(self):
        """Remove the AmlCompute object from its associated workspace.

        .. remarks::

            If this object was created through Azure Machine Learning,
            the corresponding cloud-based objects will also be deleted. If this object was created externally and only
            attached to the workspace, this method raises a :class:`azureml.exceptions.ComputeTargetException` and
            nothing is changed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('delete')

    def detach(self):
        """Detach is not supported for AmlCompute object. Use :meth:`delete` instead.

        :raises azureml.exceptions.ComputeTargetException:
        """
        raise ComputeTargetException('Detach is not supported for AmlCompute object. Try to use delete instead.')

    def serialize(self):
        """Convert this AmlCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this AmlCompute object.
        :rtype: dict
        """
        scale_settings = self.scale_settings.serialize() if self.scale_settings else None
        subnet_id = None
        if self.vnet_resourcegroup_name and self.vnet_name and self.subnet_name:
            subnet_id = "/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks" \
                        "/{2}/subnets/{3}".format(self.workspace.subscription_id, self.vnet_resourcegroup_name,
                                                  self.vnet_name, self.subnet_name)
        identity = None
        if self.identity_type:
            identity = {'type': self.identity_type, 'userAssignedIdentities': self.identity_id}
        amlcompute_properties = {'vmSize': self.vm_size, 'vmPriority': self.vm_priority,
                                 'scaleSettings': scale_settings,
                                 'userAccountCredentials': {'adminUserName': self.admin_username,
                                                            'adminUserPassword': self.admin_user_password,
                                                            'adminUserSshPublicKey': self.admin_user_ssh_key},
                                 'subnet': {'id': subnet_id},
                                 'remoteLoginPortPublicAccess': self.remote_login_port_public_access}
        amlcompute_status = self.status.serialize() if self.status else None
        cluster_properties = {'description': self.description, 'resourceId': self.cluster_resource_id,
                              'computeType': self.type, 'computeLocation': self.cluster_location,
                              'provisioningState': self.provisioning_state,
                              'provisioningErrors': self.provisioning_errors,
                              'properties': amlcompute_properties, 'status': amlcompute_status}
        return {'id': self.id, 'name': self.name, 'location': self.location, 'tags': self.tags, 'identity': identity,
                'properties': cluster_properties}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into an AmlCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the AmlCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to an AmlCompute object.
        :type object_dict: dict
        :return: The AmlCompute representation of the provided JSON object.
        :rtype: azureml.core.compute.amlcompute.AmlCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        AmlCompute._validate_get_payload(object_dict)
        target = AmlCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != AmlCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(AmlCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))
        if payload['properties']['properties']:
            for amlcompute_key in ['vmPriority', 'vmSize', 'scaleSettings']:
                if amlcompute_key not in payload['properties']['properties']:
                    raise ComputeTargetException('Invalid cluster payload, missing '
                                                 '["properties"]["properties"]["{}"]:\n'
                                                 '{}'.format(amlcompute_key, payload))

    def get(self):
        """Return compute object."""
        return ComputeTarget._get(self.workspace, self.name)

    def update(self, min_nodes=None, max_nodes=None, idle_seconds_before_scaledown=None):
        """Update the :class:`azureml.core.compute.amlcompute.ScaleSettings` for this AmlCompute target.

        :param min_nodes: The minimum number of nodes to use on the cluster.
        :type min_nodes: int
        :param max_nodes: The maximum number of nodes to use on the cluster.
        :type max_nodes: int
        :param idle_seconds_before_scaledown: The node idle time in seconds before scaling down the cluster.
        :type idle_seconds_before_scaledown: int
        """
        if min_nodes is None and max_nodes is None and idle_seconds_before_scaledown is None:
            return
        else:
            self.refresh_state()
            if min_nodes is None:
                min_nodes = self.scale_settings.minimum_node_count
            if max_nodes is None:
                max_nodes = self.scale_settings.maximum_node_count
            if idle_seconds_before_scaledown is None:
                idle_seconds_before_scaledown = self.scale_settings.idle_seconds_before_scaledown
            self.scale_settings = ScaleSettings(min_nodes, max_nodes, idle_seconds_before_scaledown)

            compute_update_payload = {'properties': {'computeType': 'AmlCompute',
                                                     'properties': {'scaleSettings': {}}}}
            compute_update_payload['properties']['properties']['scaleSettings'] = self.scale_settings.serialize()

            endpoint = self._get_compute_endpoint(self.workspace, self.name)
            headers = self.workspace._auth.get_authentication_header()
            ComputeTarget._add_request_tracking_headers(headers)
            params = {'api-version': MLC_WORKSPACE_API_VERSION}
            resp = ClientBase._execute_func(get_requests_session().patch, endpoint, params=params, headers=headers,
                                            json=compute_update_payload)

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError:
                raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                             'Response Code: {}\n'
                                             'Headers: {}\n'
                                             'Content: {}'.format(resp.status_code, resp.headers, resp.content))

            if 'Azure-AsyncOperation' in resp.headers:
                # Patch API will not return Azure-AsyncOperation if nothing has changed
                self._operation_endpoint = resp.headers['Azure-AsyncOperation']

    @staticmethod
    def _validate_identity(identity_type, identity_id):
        """Validate identity_type and identity_id fields.

        :param identity_type: Possible values are:

            * SystemAssigned - System assigned identity
            * UserAssigned - User assigned identity. Requires identity id to be set.

        :type identity_type: string
        :param identity_id: List of resource ids for the user assigned identity.
            eg. ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity
            /userAssignedIdentities/<id>']
        :type identity_id: builtin.list[str]
        """
        if identity_type not in ["UserAssigned", "SystemAssigned"]:
            raise ComputeTargetException("Invalid configuration, identity_type must be set to "
                                         "'UserAssigned' or 'SystemAssigned'.")
        elif identity_type == "UserAssigned" and (type(identity_id) is not list or not identity_id):
            raise ComputeTargetException("Invalid configuration, please pass the identity_id of the user assigned "
                                         "identity as a list (eg. ['resource_ID1','resourceID2',...])")
        elif identity_type != "UserAssigned" and identity_id:
            raise ComputeTargetException('Invalid configuration, identity_id cannot be set when '
                                         'identity type is not user assigned.')

    def add_identity(self, identity_type, identity_id=None):
        """Add Identity Type and/or Identity Ids for this AmlCompute target.

        .. remarks::

            identity_id should only be specified when identity_type == UserAssigned

        :param identity_type: Possible values are:

            * SystemAssigned - System assigned identity
            * UserAssigned - User assigned identity. Requires identity id to be set.

        :type identity_type: string
        :param identity_id: List of resource ids for the user assigned identity.
            eg. ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity
            /userAssignedIdentities/<id>']
        :type identity_id: builtin.list[str]
        """
        AmlCompute._validate_identity(identity_type, identity_id)
        self.refresh_state()
        compute_update_payload = {"location": self.location, 'properties': {}, 'identity': {}}
        if identity_id and identity_type == "UserAssigned":
            compute_update_payload['identity'] = {"type": "UserAssigned", "userAssignedIdentities": {}}
            AmlCompute._build_identity_id_payload(compute_update_payload, identity_id)
        elif identity_type == "SystemAssigned":
            compute_update_payload['identity'] = {"type": "SystemAssigned"}

        endpoint = self._get_compute_endpoint(self.workspace, self.name)
        headers = self.workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().patch, endpoint, params=params, headers=headers,
                                        json=compute_update_payload)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        identity_prop = self.get()['identity']
        self.identity_type = identity_prop['type']
        self.identity_id = \
            identity_prop['userAssignedIdentities'] if self.identity_type == "UserAssigned" else None

    def remove_identity(self, identity_id=None):
        """Remove the identity on the compute.

        .. remarks::

            System Assigned identity will be removed automatically if identity_id is not specified

        :param identity_id: User assigned identities
        :type identity_id: builtin.list[str]
        """
        if self.identity_type is None:
            raise ComputeTargetException("Cannot remove identity. The compute has no identity assigned.")

        compute_update_payload = {"location": self.location, "properties": {}, "identity": {}}
        if identity_id is None:
            compute_update_payload['identity'] = {"type": "None"}
        else:
            compute_update_payload['identity'] = {"type": "UserAssigned", "userAssignedIdentities": {}}
            for id in identity_id:
                compute_update_payload['identity']['userAssignedIdentities'][id] = None

        compute_update_payload = json.dumps(compute_update_payload)
        endpoint = self._get_compute_endpoint(self.workspace, self.name)
        headers = self.workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().patch, endpoint, params=params, headers=headers,
                                        json=compute_update_payload)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        identity_prop = self.get()['identity']
        self.identity_type = identity_prop['type']
        self.identity_id = \
            identity_prop['userAssignedIdentities'] if self.identity_type == "UserAssigned" else None

    def list_nodes(self):
        """Get the details (e.g., IP address, port, etc.) of all the compute nodes in the compute target.

        :return: The details of all the compute nodes in the compute target.
        :rtype: builtin.list
        """
        paginated_results = []
        vm_size_fmt = '{}/listNodes'
        compute_endpoint = self._get_compute_endpoint(self.workspace, self.name)
        endpoint = vm_size_fmt.format(compute_endpoint)
        headers = self.workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().post, endpoint, params=params, headers=headers)

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
        result_list = json.loads(content)
        paginated_results = get_paginated_compute_nodes(result_list, headers)

        return paginated_results

    def get_active_runs(self, type=None, tags=None, properties=None, status=None):
        """Return a generator of the runs for this compute.

        :param type: Filter the returned generator of runs by the provided type. See
            :func:`azureml.core.Run.add_type_provider` for creating run types.
        :type type: str
        :param tags: Filter runs by "tag" or {"tag": "value"}
        :type tags: str or dict
        :param properties: Filter runs by "property" or {"property": "value"}
        :type properties: str or dict
        :param status: Run status - either "Running" or "Queued"
        :type status: str
        :return: a generator of ~_restclient.models.RunDto
        :rtype: builtin.generator
        """
        workspace_client = WorkspaceClient(self.workspace.service_context)
        return workspace_client.get_runs_by_compute(self.name,
                                                    0,
                                                    DEFAULT_PAGE_SIZE,
                                                    None,
                                                    None,
                                                    type=type,
                                                    tags=tags,
                                                    properties=properties,
                                                    status=status)

    @staticmethod
    def supported_vmsizes(workspace, location=None):
        """List the supported VM sizes in a region.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param location: The location of cluster. If not specified, will default to workspace location.
        :type location: str
        :return: A list of supported VM sizes in a region with names of the VM, VCPUs, and RAM.
        :rtype: builtin.list
        """
        paginated_results = []
        if not workspace:
            return paginated_results

        if not location:
            location = workspace.location

        vm_size_fmt = '{}/subscriptions/{}/providers/Microsoft.MachineLearningServices/locations/{}/vmSizes'
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace).rstrip('/')
        endpoint = vm_size_fmt.format(resource_manager_endpoint, workspace.subscription_id, location)
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
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        result_list = json.loads(content)
        paginated_results = get_paginated_compute_supported_vms(result_list, headers)

        return paginated_results

    @staticmethod
    def list_usages(workspace, show_all=False, location=None):
        """Get the current usage information as well as limits for AML resources for given workspace and subscription.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param show_all: Specifies if detailed usages of child resources are required. Defaults to False
        :param location: The location of the resources. If not specified, will default to workspace location.
        :type location: str
        :return: List of current usage information as well as limits for AML resources
        :rtype: builtin.list
        """
        paginated_results = []
        if not workspace:
            return paginated_results

        if not location:
            location = workspace.location

        expand_children = "False"
        if show_all:
            expand_children = "True"

        usages_fmt = '{}/subscriptions/{}/providers/Microsoft.MachineLearningServices/locations/{}/usages'
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace).rstrip('/')
        endpoint = usages_fmt.format(resource_manager_endpoint, workspace.subscription_id, location)
        headers = workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION, 'expandChildren': expand_children}
        resp = ClientBase._execute_func(get_requests_session().get, endpoint, params=params, headers=headers)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Error occurred retrieving targets:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        result_list = json.loads(content)
        paginated_results = get_paginated_compute_results(result_list, headers)

        return paginated_results

    @staticmethod
    def list_quotas(workspace, location=None):
        """Get the currently assigned Workspace quotas based on VMFamily for given workspace and subscription.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param location: The location of the quotas. If not specified, will default to workspace location.
        :type location: str
        :return: List of currently assigned Workspace Quotas based on VMFamily
        :rtype: builtin.list
        """
        paginated_results = []
        if not workspace:
            return paginated_results

        if not location:
            location = workspace.location

        quotas_fmt = '{}/subscriptions/{}/providers/Microsoft.MachineLearningServices/locations/{}/quotas'
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace).rstrip('/')
        endpoint = quotas_fmt.format(resource_manager_endpoint, workspace.subscription_id, location)
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
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        result_list = json.loads(content)
        paginated_results = get_paginated_compute_results(result_list, headers)

        return paginated_results

    @staticmethod
    def update_quotas(workspace, vm_family, limit=None, location=None):
        """Update quota for a VM family in workspace.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param vm_family: VM family name
        :type vm_family: str
        :param limit: The maximum permitted quota of the resource
        :type limit: int
        :param location: The location of the quota. If not specified, will default to workspace location.
        :type location: str
        """
        if not workspace or not vm_family:
            print('Missing input parameter\n')
            return

        if not location:
            location = workspace.location

        update_quota_fmt = '{}/subscriptions/{}/providers/Microsoft.MachineLearningServices/locations/{}/updateQuotas'
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace).rstrip('/')
        endpoint = update_quota_fmt.format(resource_manager_endpoint, workspace.subscription_id, location)
        headers = workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}

        quota_id_fmt = '/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/' \
                       'workspaces/{}/quotas/{}'
        quota_id = quota_id_fmt.format(workspace.subscription_id, workspace.resource_group, workspace.name, vm_family)
        quota_body = [{'id': quota_id,
                       'type': 'Microsoft.MachineLearningServices/workspaces/quotas',
                       'limit': limit,
                       'location': workspace.location,
                       'unit': 'Count'}]

        resp = ClientBase._execute_func(get_requests_session().post, endpoint, params=params, headers=headers,
                                        json={'value': quota_body})

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
        payload = json.loads(content)

        if 'value' not in payload:
            raise ComputeTargetException('Error, invalid response payload, missing "value":\n'
                                         '{}'.format(payload))

        if len(payload['value']) > 0:
            if 'status' not in payload['value'][0]:
                raise ComputeTargetException('Error, invalid response payload, missing "status"\n'
                                             '{}'.format(payload))

            if payload['value'][0]['status'] != 'success':
                print('Could not update quota\n'
                      'Response Payload: {}'.format(payload))
            else:
                print('Updated quota\n')

        return


class AmlComputeProvisioningConfiguration(ComputeTargetProvisioningConfiguration):
    """Represents configuration parameters for provisioning AmlCompute targets.

    Use the ``provisioning_configuration`` method of the
    :class:`azureml.core.compute.AmlCompute` class to specify configuration parameters.

    :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
        Note that not all sizes are available in all regions, as
        detailed in the previous link. If not specified, defaults to Standard_NC6.
    :type vm_size: str
    :param vm_priority: The VM priority, either "dedicated" or "lowpriority" VMs.
        If not specified, defaults to "dedicated".
    :type vm_priority: str
    :param min_nodes: The minimum number of nodes to use on the cluster. If not specified, defaults to 0.
    :type min_nodes: int
    :param max_nodes: The maximum number of nodes to use on the cluster. Defaults to 4.
    :type max_nodes: int
    :param idle_seconds_before_scaledown: The node idle time in seconds before scaling down the cluster.
        If not specified, defaults to 1800.
    :type idle_seconds_before_scaledown: int
    :param admin_username: The name of the administrator user account which can be used to SSH into nodes.
    :type admin_username: str
    :param admin_user_password: The password of the administrator user account.
    :type admin_user_password: str
    :param admin_user_ssh_key: The SSH public key of the administrator user account.
    :type admin_user_ssh_key: str
    :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
    :type vnet_resourcegroup_name: str
    :param vnet_name: The name of the virtual network.
    :type vnet_name: str
    :param subnet_name: The name of the subnet inside the VNet.
    :type subnet_name: str
    :param tags: A dictionary of key value tags to provide to the compute object.
    :type tags: dict[str, str]
    :param description: A description to provide to the compute object.
    :type description: str
    :param remote_login_port_public_access: The state of the public SSH port. Possible values are:

        * Disabled - Indicates that the public ssh port is closed on all nodes of the cluster.
        * Enabled - Indicates that the public ssh port is open on all nodes of the cluster.
        * NotSpecified - Indicates that the public ssh port is closed on all nodes of the cluster if
          VNet is defined, else is open all public nodes. It can be this default value only during
          cluster creation time. After creation, it will be either enabled or disabled.

    :type remote_login_port_public_access: str
    :param identity_type: Possible values are:

            * SystemAssigned - System assigned identity
            * UserAssigned - User assigned identity. Requires identity id to be set.

    :type identity_type: string
    :param identity_id: List of resource ids for the user assigned identity.
        eg. ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity
        /userAssignedIdentities/<id>']
    :type identity_id: builtin.list[str]
    """

    def __init__(self, vm_size='', vm_priority="dedicated", min_nodes=0,
                 max_nodes=None, idle_seconds_before_scaledown=1800,
                 admin_username=None, admin_user_password=None, admin_user_ssh_key=None,
                 vnet_resourcegroup_name=None, vnet_name=None, subnet_name=None, tags=None,
                 description=None, remote_login_port_public_access="NotSpecified",
                 identity_type=None, identity_id=None, compute_location=None, enable_node_public_ip=True):
        """Create a configuration object for provisioning a AmlCompute target.

        :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
            Note that not all sizes are available in all regions, as
            detailed in the previous link. If not specified, defaults to Standard_NC6.
        :type vm_size: str
        :param vm_priority: The VM priority, either "dedicated" or "lowpriority" VMs.
            If not specified, defaults to "dedicated".
        :type vm_priority: str
        :param min_nodes: The minimum number of nodes to use on the cluster. If not specified, defaults to 0.
        :type min_nodes: int
        :param max_nodes: The maximum number of nodes to use on the cluster. Defaults to 4.
        :type max_nodes: int
        :param idle_seconds_before_scaledown: The node idle time in seconds before scaling down the cluster.
            If not specified, defaults to 1800.
        :type idle_seconds_before_scaledown: int
        :param admin_username: The name of the administrator user account which can be used to SSH into nodes.
        :type admin_username: str
        :param admin_user_password: The password of the administrator user account.
        :type admin_user_password: str
        :param admin_user_ssh_key: The SSH public key of the administrator user account.
        :type admin_user_ssh_key: str
        :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
        :type vnet_resourcegroup_name: str
        :param vnet_name: The name of the virtual network.
        :type vnet_name: str
        :param subnet_name: The name of the subnet inside the VNet.
        :type subnet_name: str
        :param tags: A dictionary of key value tags to provide to the compute object.
        :type tags: dict[str, str]
        :param description: A description to provide to the compute object.
        :type description: str
        :param remote_login_port_public_access: The state of the public SSH port. Possible values are:

            * Disabled - Indicates that the public ssh port is closed on all nodes of the cluster.
            * Enabled - Indicates that the public ssh port is open on all nodes of the cluster.
            * NotSpecified - Indicates that the public ssh port is closed on all nodes of the cluster if
              VNet is defined, else is open all public nodes. This is the default value. The state
              can be in this default value only during cluster creation time. After creation, it will be either
              enabled or disabled.

        :type remote_login_port_public_access: str
        :param identity_type: Possible values are:

            * SystemAssigned - System assigned identity
            * UserAssigned - User assigned identity. Requires identity id to be set.

        :type identity_type: string
        :param identity_id: List of resource ids for the user assigned identity.
            eg. ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity
            /userAssignedIdentities/<id>']
        :type identity_id: builtin.list[str]
        :param compute_location: Location to provision cluster in.
        :type compute_location: str
        :param enable_node_public_ip: Enable node public IP. Possible values are:

            * True - Enable node public IP.
            * False - Disable node public IP.
            * NotSpecified - Enable node public IP.
        :type enable_node_public_ip: bool
        :return: A configuration object to be used when creating a Compute object.
        :rtype: azureml.core.compute.amlcompute.AmlComputeProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        super(AmlComputeProvisioningConfiguration, self).__init__(AmlCompute, None)
        self.vm_size = vm_size
        self.vm_priority = vm_priority
        min_nodes = 0 if not min_nodes else min_nodes
        self.scale_settings = ScaleSettings(min_nodes, max_nodes, idle_seconds_before_scaledown)
        self.admin_username = admin_username
        self.admin_user_password = admin_user_password
        self.admin_user_ssh_key = admin_user_ssh_key
        self.vnet_resourcegroup_name = vnet_resourcegroup_name
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name
        self.tags = tags
        self.description = description
        self.remote_login_port_public_access = remote_login_port_public_access
        self.identity_type = identity_type
        self.identity_id = identity_id
        self.compute_location = compute_location
        self.enable_node_public_ip = enable_node_public_ip
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        if any([self.admin_username, self.admin_user_password, self.admin_user_ssh_key]) and \
                not self.admin_username:
            raise ComputeTargetException('Invalid user credentials, admin username must be provided along '
                                         'with either a password or SSH key information.')
        if self.admin_username:
            if not self.admin_user_password and not self.admin_user_ssh_key:
                raise ComputeTargetException('Invalid user credentials, no password or key information provided. '
                                             'Please provide either a password or SSH key information.')
            elif self.admin_user_password and not self._is_valid_password():
                # This error message comes from the work item
                # https://msdata.visualstudio.com/Vienna/_workitems/edit/825399/ and its screenshot of the UX team's
                # admin user password validation error message
                raise ComputeTargetException('Invalid admin user password provided. The admin user password must have '
                                             '3 of the following: 1 lowercase character, 1 uppercase character, 1 '
                                             'number, and 1 special character `~!@#$%^&*()=+_[]{}|;:./\'",<>?')
        if any([self.vnet_name, self.vnet_resourcegroup_name, self.subnet_name]) and \
                not all([self.vnet_name, self.vnet_resourcegroup_name, self.subnet_name]):
            raise ComputeTargetException('Invalid configuration, not all virtual net information provided. '
                                         'To use a custom virtual net, please provide vnet name, vnet resource '
                                         'group and subnet name')
        if self.scale_settings.maximum_node_count is None:
            raise ComputeTargetException('Invalid provisioning configuration, max nodes count is mandatory')
        elif self.scale_settings.maximum_node_count < 0:
            raise ComputeTargetException('Invalid provisioning configuration, max nodes count must be greater '
                                         'than or equal to 0')
        elif self.scale_settings.maximum_node_count < self.scale_settings.minimum_node_count:
            raise ComputeTargetException('Invalid provisioning configuration, max nodes count must be greater than or '
                                         'equal to min nodes count')
        if self.identity_type:
            AmlCompute._validate_identity(self.identity_type, self.identity_id)
        elif self.identity_id:
            raise ComputeTargetException('Invalid configuration, identity_id cannot be set when '
                                         'identity type is not user assigned.')

    def _is_valid_password(self) -> bool:
        """Check if the provides password is valid according to the password policy.

        :return: True if the password is valid. False otherwise.
        """
        if self.admin_user_password is None:
            return False
        password_characters = set(self.admin_user_password)

        def contains(pwd_set: set, char_class: set) -> bool:
            return len(pwd_set & char_class) != 0

        satisfied_categories = 0
        permittedSpecialCharacters = set("`~!@#$%^&*()=+_[]{}|;:./'\",<>?")
        if contains(password_characters, set(string.ascii_lowercase)):
            satisfied_categories += 1
        if contains(password_characters, set(string.ascii_uppercase)):
            satisfied_categories += 1
        if contains(password_characters, set(string.digits)):
            satisfied_categories += 1
        if contains(password_characters, permittedSpecialCharacters):
            satisfied_categories += 1

        # Three required categories is needed to match what UX requires
        return satisfied_categories >= 3


class ScaleSettings(object):
    """Represents scale settings for an AmlCompute target.

    Use the :class:`azureml.core.compute.AmlCompute` class ``provisioning_configuration`` method to specify scale
    settings, the ``update`` method to update them, and the ``get_status`` method to view them.
    """

    def __init__(self, minimum_node_count, maximum_node_count, idle_seconds_before_scaledown):
        """Initialize the ScaleSettings object.

        :param minimum_node_count: The minimum number of nodes to use on the cluster.
        :type minimum_node_count: int
        :param maximum_node_count: The maximum number of nodes to use on the cluster.
        :type maximum_node_count: int
        :param idle_seconds_before_scaledown: The node idle time in seconds before scaling down the cluster.
        :type idle_seconds_before_scaledown: int
        """
        self.minimum_node_count = minimum_node_count
        self.maximum_node_count = maximum_node_count
        self.idle_seconds_before_scaledown = idle_seconds_before_scaledown

    def serialize(self):
        """Convert this ScaleSettings object into a JSON serialized dictionary.

        :return: The JSON representation of this ScaleSettings object.
        :rtype: dict
        """
        duration = ""
        if self.idle_seconds_before_scaledown:
            duration = convert_seconds_to_duration(self.idle_seconds_before_scaledown)

        return {'minNodeCount': self.minimum_node_count,
                'maxNodeCount': self.maximum_node_count,
                'nodeIdleTimeBeforeScaleDown': duration}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into a ScaleSettings object.

        :param object_dict: A JSON object to convert to a ScaleSettings object.
        :type object_dict: dict
        :return: The ScaleSettings representation of the provided JSON object.
        :rtype: azureml.core.compute.amlcompute.ScaleSettings
        :raises azureml.exceptions.ComputeTargetException:
        """
        if not object_dict:
            return None
        for key in ['minNodeCount', 'maxNodeCount', 'nodeIdleTimeBeforeScaleDown']:
            if key not in object_dict:
                raise ComputeTargetException('Invalid scale settings payload, missing "{}":\n'
                                             '{}'.format(key, object_dict))

        return ScaleSettings(object_dict['minNodeCount'],
                             object_dict['maxNodeCount'],
                             convert_duration_to_seconds(object_dict['nodeIdleTimeBeforeScaleDown']))


class AmlComputeStatus(object):
    """Represents detailed status information about an AmlCompute target.

    Use the :meth:`azureml.core.compute.AmlCompute.get_status` method of the :class:`azureml.core.compute.AmlCompute`
    class to return status information.

    :param allocation_state: A string description of the current allocation state.
    :type allocation_state: str
    :param allocation_state_transition_time: The time of the most recent allocation state change.
    :type allocation_state_transition_time: datetime.datetime
    :param creation_time: The cluster creation time.
    :type creation_time: datetime.datetime
    :param current_node_count: The current number of nodes used by the cluster.
    :type current_node_count: int
    :param errors: A list of error details, if any.
    :type errors: builtin.list
    :param modified_time: The cluster modification time.
    :type modified_time: datetime.datetime
    :param node_state_counts: An object containing counts of the various current node states in the cluster.
    :type node_state_counts: azureml.core.compute.amlcompute.AmlComputeNodeStateCounts
    :param provisioning_state: The current provisioning state of the cluster.
    :type provisioning_state: str
    :param provisioning_state_transition_time: The time of the most recent provisioning state change.
    :type provisioning_state_transition_time: datetime.datetime
    :param scale_settings: An object containing the specified scale settings for the cluster.
    :type scale_settings: azureml.core.compute.amlcompute.ScaleSettings
    :param target_node_count: The target number of nodes for by the cluster.
    :type target_node_count: int
    :param vm_priority: The VM priority. Can be "dedicated" (default) or "lowpriority". Low Priority VMs use
        Azure's excess capacity are cheapter but run the risk of being pre-empted.
    :type vm_priority: str
    :param vm_size: The size of the agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
        Note that not all sizes are available in all regions, as detailed in the previous link.
    :type vm_size: str
    """

    def __init__(self, allocation_state, allocation_state_transition_time, creation_time, current_node_count,
                 errors, modified_time, node_state_counts, provisioning_state, provisioning_state_transition_time,
                 scale_settings, target_node_count, vm_priority, vm_size):
        """Initialize a AmlComputeStatus object.

        :param allocation_state: A string description of the current allocation state.
        :type allocation_state: str
        :param allocation_state_transition_time: The time of the most recent allocation state change.
        :type allocation_state_transition_time: datetime.datetime
        :param creation_time: The cluster creation time.
        :type creation_time: datetime.datetime
        :param current_node_count: The current number of nodes used by the cluster.
        :type current_node_count: int
        :param errors: A list of error details, if any.
        :type errors: builtin.list
        :param modified_time: The cluster modification time.
        :type modified_time: datetime.datetime
        :param node_state_counts: An object containing counts of the various current node states in the cluster.
        :type node_state_counts: azureml.core.compute.amlcompute.AmlComputeNodeStateCounts
        :param provisioning_state: The current provisioning state of the cluster.
        :type provisioning_state: str
        :param provisioning_state_transition_time: The time of the most recent provisioning state change.
        :type provisioning_state_transition_time: datetime.datetime
        :param scale_settings: An object containing the specified scale settings for the cluster.
        :type scale_settings: azureml.core.compute.amlcompute.ScaleSettings
        :param target_node_count: The target number of nodes for by the cluster.
        :type target_node_count: int
        :param vm_priority: The VM priority. Can be "dedicated" (default) or "lowpriority". Low Priority VMs use
            Azure's excess capacity are cheapter but run the risk of being pre-empted.
        :type vm_priority: str
        :param vm_size: The size of the agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
            Note that not all sizes are available in all regions, as detailed in the previous link.
        :type vm_size: str
        """
        self.allocation_state = allocation_state
        self.allocation_state_transition_time = allocation_state_transition_time
        self.creation_time = creation_time
        self.current_node_count = current_node_count
        self.errors = errors
        self.modified_time = modified_time
        self.node_state_counts = node_state_counts
        self.provisioning_state = provisioning_state
        self.provisioning_state_transition_time = provisioning_state_transition_time
        self.scale_settings = scale_settings
        self.target_node_count = target_node_count
        self.vm_priority = vm_priority
        self.vm_size = vm_size

    def serialize(self):
        """Convert this AmlComputeStatus object into a JSON serialized dictionary.

        :return: The JSON representation of this AmlComputeStatus object.
        :rtype: dict
        """
        allocation_state_transition_time = self.allocation_state_transition_time.isoformat() \
            if self.allocation_state_transition_time else None
        creation_time = self.creation_time.isoformat() if self.creation_time else None
        modified_time = self.modified_time.isoformat() if self.modified_time else None
        node_state_counts = self.node_state_counts.serialize() if self.node_state_counts else None
        provisioning_state_transition_time = self.provisioning_state_transition_time.isoformat() \
            if self.provisioning_state_transition_time else None
        scale_settings = self.scale_settings.serialize() if self.scale_settings else None
        return {'currentNodeCount': self.current_node_count, 'targetNodeCount': self.target_node_count,
                'nodeStateCounts': node_state_counts, 'allocationState': self.allocation_state,
                'allocationStateTransitionTime': allocation_state_transition_time, 'errors': self.errors,
                'creationTime': creation_time, 'modifiedTime': modified_time,
                'provisioningState': self.provisioning_state,
                'provisioningStateTransitionTime': provisioning_state_transition_time,
                'scaleSettings': scale_settings,
                'vmPriority': self.vm_priority, 'vmSize': self.vm_size}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into an AmlComputeStatus object.

        :param object_dict: A JSON object to convert to an AmlComputeStatus object.
        :type object_dict: dict
        :return: The AmlComputeStatus representation of the provided JSON object.
        :rtype: azureml.core.compute.amlcompute.AmlComputeStatus
        :raises azureml.exceptions.ComputeTargetException:
        """
        if not object_dict:
            return None
        allocation_state = object_dict['properties']['allocationState'] \
            if 'allocationState' in object_dict['properties'] else None
        allocation_state_transition_time = parse(object_dict['properties']['allocationStateTransitionTime']) \
            if 'allocationStateTransitionTime' in object_dict['properties'] else None
        creation_time = parse(object_dict['createdOn']) \
            if 'createdOn' in object_dict else None
        current_node_count = object_dict['properties']['currentNodeCount'] \
            if 'currentNodeCount' in object_dict['properties'] else None
        errors = object_dict['properties']['errors'] \
            if 'errors' in object_dict['properties'] else None
        modified_time = parse(object_dict['modifiedOn']) \
            if 'modifiedOn' in object_dict else None
        node_state_counts = AmlComputeNodeStateCounts.deserialize(object_dict['properties']['nodeStateCounts']) \
            if 'nodeStateCounts' in object_dict['properties'] else None
        provisioning_state = object_dict['provisioningState'] \
            if 'provisioningState' in object_dict else None
        provisioning_state_transition_time = parse(object_dict['provisioningStateTransitionTime']) \
            if 'provisioningStateTransitionTime' in object_dict else None
        scale_settings = ScaleSettings.deserialize(object_dict['properties']['scaleSettings']) \
            if 'scaleSettings' in object_dict['properties'] else None
        target_node_count = object_dict['properties']['targetNodeCount'] \
            if 'targetNodeCount' in object_dict['properties'] else None
        vm_priority = object_dict['properties']['vmPriority'] if 'vmPriority' in object_dict['properties'] else None
        vm_size = object_dict['properties']['vmSize'] if 'vmSize' in object_dict['properties'] else None

        return AmlComputeStatus(allocation_state, allocation_state_transition_time, creation_time, current_node_count,
                                errors, modified_time, node_state_counts, provisioning_state,
                                provisioning_state_transition_time, scale_settings, target_node_count, vm_priority,
                                vm_size)


class AmlComputeNodeStateCounts(object):
    """Represents detailed node count information for an AmlCompute object.

    Use the :meth:`azureml.core.compute.AmlCompute.get_status` method of the :class:`azureml.core.compute.AmlCompute`
    to view node counts.

    :param idle_node_count: The current number of idle nodes.
    :type idle_node_count: int
    :param leaving_node_count: The current number of nodes that are being deprovisioned.
    :type leaving_node_count: int
    :param preempted_node_count: The current number of preempted nodes.
    :type preempted_node_count: int
    :param preparing_node_count: The current number of nodes that are being provisioned.
    :type preparing_node_count: int
    :param running_node_count: The current number of in use nodes.
    :type running_node_count: int
    :param unusable_node_count: The current number of unusable nodes.
    :type unusable_node_count: int
    """

    def __init__(self, idle_node_count, leaving_node_count, preempted_node_count, preparing_node_count,
                 running_node_count, unusable_node_count):
        """Initialize a AmlComputeNodeStateCounts object.

        :param idle_node_count: Current number of idle nodes
        :type idle_node_count: int
        :param leaving_node_count: Current number of nodes that are being deprovisioned
        :type leaving_node_count: int
        :param preempted_node_count: Current number of preempted nodes
        :type preempted_node_count: int
        :param preparing_node_count: Current number of nodes that are being provisioned
        :type preparing_node_count: int
        :param running_node_count: Current number of in use nodes
        :type running_node_count: int
        :param unusable_node_count: Current number of unusable nodes
        :type unusable_node_count: int
        """
        self.idle_node_count = idle_node_count
        self.leaving_node_count = leaving_node_count
        self.preempted_node_count = preempted_node_count
        self.preparing_node_count = preparing_node_count
        self.running_node_count = running_node_count
        self.unusable_node_count = unusable_node_count

    def serialize(self):
        """Convert this AmlComputeNodeStateCounts object into a JSON serialized dictionary.

        :return: The JSON representation of this AmlComputeNodeStateCounts object.
        :rtype: dict
        """
        return {'preparingNodeCount': self.preparing_node_count, 'runningNodeCount': self.running_node_count,
                'idleNodeCount': self.idle_node_count, 'unusableNodeCount': self.unusable_node_count,
                'leavingNodeCount': self.leaving_node_count, 'preemptedNodeCount': self.preempted_node_count}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into an AmlComputeNodeStateCounts object.

        :param object_dict: A JSON object to convert to an AmlComputeNodeStateCounts object.
        :type object_dict: dict
        :return: The AmlComputeNodeStateCounts representation of the provided JSON object.
        :rtype: azureml.core.compute.amlcompute.AmlComputeNodeStateCounts
        :raises azureml.exceptions.ComputeTargetException:
        """
        if not object_dict:
            return None
        idle_node_count = object_dict['idleNodeCount'] if 'idleNodeCount' in object_dict else None
        leaving_node_count = object_dict['leavingNodeCount'] if 'leavingNodeCount' in object_dict else None
        preempted_node_count = object_dict['preemptedNodeCount'] if 'preemptedNodeCount' in object_dict else None
        preparing_node_count = object_dict['preparingNodeCount'] if 'preparingNodeCount' in object_dict else None
        running_node_count = object_dict['runningNodeCount'] if 'runningNodeCount' in object_dict else None
        unusable_node_count = object_dict['unusableNodeCount'] if 'unusableNodeCount' in object_dict else None

        return AmlComputeNodeStateCounts(idle_node_count, leaving_node_count, preempted_node_count,
                                         preparing_node_count, running_node_count, unusable_node_count)
