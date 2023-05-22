# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for creating a fully-managed cloud-based workstation in Azure Machine Learning."""

import copy
import json
import requests
import sys
import time
import logging

from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml._compute._constants import GRAPH_API_VERSION
from azureml._compute._util import get_paginated_compute_supported_vms
from azureml._compute._util import get_requests_session
from azureml._compute._util import computeinstance_payload_template
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetProvisioningConfiguration
from azureml.exceptions import ComputeTargetException, AuthenticationException
from azureml._restclient.clientbase import ClientBase
from azureml._restclient.workspace_client import WorkspaceClient
from dateutil.parser import parse
from collections import OrderedDict
from azureml.core._portal import HasWorkspacePortal
from azureml.core._docs import get_docs_url
from azureml._html.utilities import to_html, make_link


module_logger = logging.getLogger(__name__)


class ComputeInstance(ComputeTarget):
    """Manages a cloud-based, optimized ML development environment in Azure Machine Learning.

    An Azure Machine Learning compute instance is a fully-configured and managed development environment
    in the cloud that is optimized for machine learning development workflows. ComputeInstance is typically
    used to create a development environment or as a compute target for training and inference for development
    and testing. With a ComputeInstance you can author, train, and deploy models in a fully integrated notebook
    experience in your workspace. For more information, see `What is an Azure Machine Learning compute
    instance? <https://docs.microsoft.com/azure/machine-learning/concept-compute-instance>`_.
    """

    _compute_type = 'ComputeInstance'
    _user_display_info = dict()

    def _initialize(self, workspace, obj_dict):
        """Initialize a ComputeInstance object.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param obj_dict: A dictionary of ComputeInstance properties.
        :type obj_dict: dict
        :return: None
        :rtype: None
        """
        name = obj_dict['name']
        compute_resource_id = MLC_COMPUTE_RESOURCE_ID_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                                 workspace.name, name)
        resource_manager_endpoint = self._get_resource_manager_endpoint(workspace)
        mlc_endpoint = '{}{}'.format(resource_manager_endpoint, compute_resource_id)
        location = obj_dict['location']
        compute_type = obj_dict['properties']['computeType']
        tags = obj_dict['tags']
        description = obj_dict['properties']['description']
        created_on = obj_dict['properties']['createdOn']
        modified_on = obj_dict['properties']['modifiedOn']
        resource_id = obj_dict['properties']['resourceId']
        location = obj_dict['properties']['computeLocation'] \
            if 'computeLocation' in obj_dict['properties'] else None
        provisioning_state = obj_dict['properties']['provisioningState']
        provisioning_errors = obj_dict['properties']['provisioningErrors']
        is_attached = obj_dict['properties']['isAttachedCompute']
        vm_size = obj_dict['properties']['properties']['vmSize'] \
            if obj_dict['properties']['properties'] else None
        ssh_settings = obj_dict['properties']['properties']['sshSettings'] \
            if obj_dict['properties']['properties'] and \
            'sshSettings' in obj_dict['properties']['properties'] else None
        admin_username = ssh_settings['adminUserName'] \
            if ssh_settings and 'adminUserName' in ssh_settings else None
        admin_user_ssh_key = ssh_settings['adminPublicKey'] \
            if ssh_settings and 'adminPublicKey' in ssh_settings else None
        ssh_public_access = (ssh_settings['sshPublicAccess'] == "Enabled") \
            if ssh_settings and 'sshPublicAccess' in ssh_settings else False
        ssh_port = ssh_settings['sshPort'] \
            if ssh_settings and 'sshPort' in ssh_settings else None
        personal_compute_instance_settings = obj_dict['properties']['properties']['personalComputeInstanceSettings'] \
            if obj_dict['properties']['properties'] and \
            'personalComputeInstanceSettings' in obj_dict['properties']['properties'] else None
        assigned_user = personal_compute_instance_settings['assignedUser'] \
            if personal_compute_instance_settings and 'assignedUser' in personal_compute_instance_settings else None
        assigned_user_object_id = assigned_user['objectId'] \
            if assigned_user and 'objectId' in assigned_user else None
        assigned_user_tenant_id = assigned_user['tenantId'] \
            if assigned_user and 'tenantId' in assigned_user else None
        public_ip_address = obj_dict['properties']['properties']['connectivityEndpoints']['publicIpAddress'] \
            if obj_dict['properties']['properties'] and \
            'connectivityEndpoints' in obj_dict['properties']['properties'] and \
            obj_dict['properties']['properties']['connectivityEndpoints'] and \
            'publicIpAddress' in obj_dict['properties']['properties']['connectivityEndpoints'] else None
        private_ip_address = obj_dict['properties']['properties']['connectivityEndpoints']['privateIpAddress'] \
            if obj_dict['properties']['properties'] and \
            'connectivityEndpoints' in obj_dict['properties']['properties'] and \
            obj_dict['properties']['properties']['connectivityEndpoints'] and \
            'privateIpAddress' in obj_dict['properties']['properties']['connectivityEndpoints'] else None
        applications = obj_dict['properties']['properties']['applications'] \
            if obj_dict['properties']['properties'] and \
            'applications' in obj_dict['properties']['properties'] else None
        errors = obj_dict['properties']['properties']['errors'] \
            if obj_dict['properties']['properties'] and \
            'errors' in obj_dict['properties']['properties'] else None
        vnet_resourcegroup_name = None
        vnet_name = None
        subnet_name = None
        subnet_id = obj_dict['properties']['properties']['subnet']['id'] \
            if obj_dict['properties']['properties'] and obj_dict['properties']['properties']['subnet'] else None
        if subnet_id:
            vnet_resourcegroup_name = subnet_id[subnet_id.index("/resourceGroups/")
                                                + len("/resourceGroups/"):subnet_id.index("/providers")]
            vnet_name = subnet_id[subnet_id.index("/virtualNetworks/")
                                  + len("/virtualNetworks/"):subnet_id.index("/subnets")]
            subnet_name = subnet_id[subnet_id.index("/subnets/") + len("/subnets/"):]
        status = ComputeInstanceStatus.deserialize(obj_dict['properties'])
        super(ComputeInstance, self)._initialize(compute_resource_id, name, location, compute_type, tags, description,
                                                 created_on, modified_on, provisioning_state, provisioning_errors,
                                                 resource_id, location, workspace, mlc_endpoint, None,
                                                 workspace._auth, is_attached)

        if not status.created_by_user_name:
            status.created_by_user_name = ComputeInstance._get_user_display_name(
                self.workspace,
                status.created_by_user_id,
                status.created_by_user_org)

        self.vm_size = vm_size
        self.ssh_public_access = ssh_public_access
        self.admin_username = admin_username
        self.admin_user_ssh_public_key = admin_user_ssh_key
        self.ssh_port = ssh_port
        self.vnet_resourcegroup_name = vnet_resourcegroup_name
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name
        self.status = status
        self.public_ip_address = public_ip_address
        self.private_ip_address = private_ip_address
        self.applications = applications
        self.errors = errors
        self.subnet_id = subnet_id
        self.assigned_user_object_id = assigned_user_object_id
        self.assigned_user_tenant_id = assigned_user_tenant_id

    def __repr__(self):
        """Return the string representation of the ComputeInstance object.

        :return: String representation of the ComputeInstance object
        :rtype: str
        """
        return json.dumps(self.serialize(), indent=2)

    def _get_base_info_dict(self):
        """Return ComputeInstance base info dictionary.

        :return: ComputeInstance info dictionary
        :rtype: OrderedDict
        """
        info = OrderedDict([
            ('Name', self.name),
            ('Id', self.id),
            ('Workspace', self.workspace.name),
            ('Location', self.location),
            ('VmSize', self.vm_size),
            ('State', self._get_state()),
            ('Tags', self.tags)
        ])

        if self.ssh_public_access and self.public_ip_address and self.ssh_port and self.admin_username:
            info['SSH'] = '{}@{} -p {}'.format(self.admin_username, self.public_ip_address, self.ssh_port)

        if self.subnet_id:
            info['Subnet'] = self.subnet_id

        return info

    @classmethod
    def get_docs_url(cls):
        """Url to the documentation for this class.

        :return: url
        :rtype: str
        """
        return get_docs_url(cls)

    def __str__(self):
        """Format ComputeInstance data into a string.

        :return: Return the string form of the ComputeInstance object
        :rtype: str
        """
        return json.dumps(self._get_base_info_dict(), indent=2)

    def _repr_html_(self):
        """Html representation of the object.

        :return: Return an html table representing this ComputeInstance
        :rtype: str
        """
        info = OrderedDict()
        workspace_portal = HasWorkspacePortal(self.workspace)
        portal_domain_url = workspace_portal._get_portal_domain(self.workspace)
        compute_url = '{}/compute/{}/details?wsid=/subscriptions/{}/resourcegroups/{}/workspaces/{}'.format(
                      portal_domain_url,
                      self.name,
                      self.workspace.subscription_id,
                      self.workspace.resource_group,
                      self.workspace.name)
        info['Name'] = make_link(compute_url, self.name)
        info['Workspace'] = make_link(workspace_portal.get_portal_url(), self.workspace.name)
        info['State'] = self._get_state()
        info['Location'] = self.location
        info['VmSize'] = self.vm_size

        if self._get_state().endswith('Running') and self.applications:
            uri = []
            for app in self.applications:
                if(app['displayName'] == 'Jupyter'):
                    uri.append(make_link(app['endpointUri'], 'Jupyter'))
                elif(app['displayName'] == 'Jupyter Lab'):
                    uri.append(make_link(app['endpointUri'], 'JupyterLab'))
                elif(app['displayName'] == 'RStudio'):
                    uri.append(make_link(app['endpointUri'], 'RStudio'))

            if len(uri) > 0:
                info['Application URI'] = '  '.join(uri)

        if self.ssh_public_access and self.public_ip_address and self.ssh_port and self.admin_username:
            info['SSH'] = '{}@{} -p {}'.format(self.admin_username, self.public_ip_address, self.ssh_port)

        info['Docs'] = make_link(self.get_docs_url(), 'Doc')
        return to_html(info)

    @staticmethod
    def _create(workspace, name, provisioning_configuration):
        """Create implementation method.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: Name of the ComputeInstance.
        :type name: str
        :param provisioning_configuration:
        :type provisioning_configuration: ComputeInstanceProvisioningConfiguration
        :return: The ComputeInstance.
        :rtype: azureml.core.compute.ComputeInstance.ComputeInstance
        """
        ComputeInstance._validate_provisioning_configuration(workspace, name, provisioning_configuration)
        compute_create_payload = ComputeInstance._build_create_payload(
            provisioning_configuration,
            workspace.location,
            workspace.subscription_id)
        return ComputeTarget._create_compute_target(workspace, name, compute_create_payload, ComputeInstance)

    @staticmethod
    def _validate_provisioning_configuration(workspace, name, provisioning_configuration):
        """Validate provisioning configuration.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: Name of the ComputeInstance.
        :type name: str
        :param provisioning_configuration:
        :type provisioning_configuration: ComputeInstanceProvisioningConfiguration
        :return: None.
        :rtype: None
        """
        vm_size = provisioning_configuration.vm_size
        supported_vmsizes = ComputeInstance.supported_vmsizes(workspace)
        available = any(vmSize.get('name').lower() == vm_size.lower() for vmSize in supported_vmsizes)
        if not available:
            raise ComputeTargetException("VmSize '{vm_size}' is not available in the specified region. "
                                         "More details can be found here: https://aka.ms/azureml-vm-details."
                                         .format(vm_size=vm_size))
        ComputeInstance._validate_name(workspace, name)

    @staticmethod
    def _validate_name(workspace, name):
        """Validate compute instance name.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param name: Name of the ComputeInstance.
        :type name: str
        :return: None.
        :rtype: None
        """
        endpoint = '{arm_endpoint}/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/' \
                   'Microsoft.MachineLearningServices/workspaces/{workspace_name}/checkComputeNameAvailability'
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace)
        endpoint = endpoint.format(
            arm_endpoint=resource_manager_endpoint,
            subscription_id=workspace.subscription_id,
            resource_group=workspace.resource_group,
            workspace_name=workspace.name)
        headers = workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        body = {'Name': name, 'Type': 'Microsoft.BatchAI/DataScienceInstance'}
        resp = ClientBase._execute_func(get_requests_session().post, endpoint, params=params, headers=headers,
                                        json=body)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException(
                'Error occurred in name validation:\n'
                'Response Code: {status_code}\n'
                'Headers: {headers}\n'
                'Content: {content}'
                .format(status_code=resp.status_code, headers=resp.headers, content=resp.content))

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = json.loads(content)
        if not content.get("nameAvailable"):
            raise ComputeTargetException(
                "Compute name '{name}' is not available. Reason: {reason}. Message: {message}"
                .format(name=name, reason=content.get("reason"), message=content.get("message", "")))

    @staticmethod
    def provisioning_configuration(vm_size='', ssh_public_access=False, admin_user_ssh_public_key=None,
                                   vnet_resourcegroup_name=None, vnet_name=None, subnet_name=None, tags=None,
                                   description=None, assigned_user_object_id=None, assigned_user_tenant_id=None):
        """Create a configuration object for provisioning a ComputeInstance target.

        :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
            Note that not all sizes are available in all regions, as
            detailed in the previous link. Defaults to Standard_NC6.
        :type vm_size: str
        :param ssh_public_access: Indicates the state of the public SSH port. Possible values are:
            * False - The public SSH port is closed.
            * True - The public SSH port is open.
        :type ssh_public_access: bool
        :param admin_user_ssh_public_key: The SSH public key of the administrator user account.
        :type admin_user_ssh_public_key: str
        :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
        :type vnet_resourcegroup_name: str
        :param vnet_name: The name of the virtual network.
        :type vnet_name: str
        :param subnet_name: The name of the subnet inside the vnet.
        :type subnet_name: str
        :param tags: An optional dictionary of key value tags to associate with the compute object.
        :type tags: dict[str, str]
        :param description: An optional description for the compute object.
        :type description: str
        :param assigned_user_object_id: The AAD Object ID of the assigned user (preview).
        :type assigned_user_object_id: str
        :param assigned_user_tenant_id: The AAD Tenant ID of the assigned user (preview).
        :type assigned_user_tenant_id: str
        :return: A configuration object to be used when creating a Compute object.
        :rtype: azureml.core.compute.computeinstance.ComputeInstanceProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        config = ComputeInstanceProvisioningConfiguration(
            vm_size, ssh_public_access, admin_user_ssh_public_key, vnet_resourcegroup_name,
            vnet_name, subnet_name, tags, description, assigned_user_object_id, assigned_user_tenant_id)
        return config

    @staticmethod
    def _build_create_payload(config, location, subscription_id):
        """Construct the payload needed to create an ComputeInstance.

        :param config: ComputeInstance provisioning configuration.
        :type config: azureml.core.compute.ComputeInstanceProvisioningConfiguration
        :param location: The location of the compute.
        :type location: str
        :param subscription_id: The subscription ID.
        :type subscription_id: str
        :return: A dictionary of ComputeInstance provisioning configuration properties.
        :rtype: dict
        """
        json_payload = copy.deepcopy(computeinstance_payload_template)
        del(json_payload['properties']['resourceId'])
        del(json_payload['properties']['computeLocation'])
        json_payload['location'] = location
        if not config.vm_size and not config.admin_user_ssh_public_key and not config.ssh_public_access and not \
                config.vnet_resourcegroup_name and not config.vnet_name and not config.subnet_name and not \
                config.assigned_user_object_id and not config.assigned_user_tenant_id:
            del(json_payload['properties']['properties'])
        else:
            if not config.vm_size:
                del(json_payload['properties']['properties']['vmSize'])
            else:
                json_payload['properties']['properties']['vmSize'] = config.vm_size
            if not config.admin_user_ssh_public_key and not isinstance(config.ssh_public_access, bool):
                del(json_payload['properties']['properties']['sshSettings'])
            else:
                if not isinstance(config.ssh_public_access, bool):
                    del(json_payload['properties']['properties']['sshSettings']['sshPublicAccess'])
                else:
                    json_payload['properties']['properties']['sshSettings']['sshPublicAccess'] = \
                        ("Enabled" if config.ssh_public_access else "Disabled")
                if not config.admin_user_ssh_public_key:
                    del(json_payload['properties']['properties']['sshSettings']['adminPublicKey'])
                else:
                    json_payload['properties']['properties']['sshSettings']['adminPublicKey'] = \
                        config.admin_user_ssh_public_key
            if not config.assigned_user_object_id and not config.assigned_user_tenant_id:
                del(json_payload['properties']['properties']['personalComputeInstanceSettings'])
            else:
                user = json_payload['properties']['properties']['personalComputeInstanceSettings']['assignedUser']
                user['objectId'] = config.assigned_user_object_id
                user['tenantId'] = config.assigned_user_tenant_id
            if not config.vnet_name:
                del(json_payload['properties']['properties']['subnet'])
            else:
                json_payload['properties']['properties']['subnet'] = \
                    {"id": "/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks"
                     "/{2}/subnets/{3}".format(subscription_id, config.vnet_resourcegroup_name,
                                               config.vnet_name, config.subnet_name)}
        if config.tags:
            json_payload['tags'] = config.tags
        else:
            del(json_payload['tags'])
        if config.description:
            json_payload['properties']['description'] = config.description
        else:
            del(json_payload['properties']['description'])
        return json_payload

    def wait_for_completion(self, show_output=False, is_delete_operation=False):
        """Wait for the ComputeInstance to finish provisioning.

        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool
        :param is_delete_operation: Indicates whether the operation is meant for deleting.
        :type is_delete_operation: bool
        :raises azureml.exceptions.ComputeTargetException:
        """
        if is_delete_operation:
            super(ComputeInstance, self).wait_for_completion(show_output=show_output,
                                                             is_delete_operation=is_delete_operation)
        else:
            state = ''
            prev_state = ''

            while True:
                time.sleep(5)
                self.refresh_state()
                state = self._get_state()

                if show_output and state:
                    if state != prev_state:
                        if prev_state is None:
                            sys.stdout.write('{}'.format(state))
                        else:
                            sys.stdout.write('\n{}'.format(state))
                    elif state:
                        sys.stdout.write('.')
                    sys.stdout.flush()

                terminal_state_reached = self._terminal_state_reached()
                status_errors_present = self._status_errors_present()

                if terminal_state_reached or status_errors_present:
                    break

                prev_state = state

            if show_output:
                sys.stdout.write('\n')
                sys.stdout.flush()

            if terminal_state_reached:
                state = self._get_state()
                if show_output:
                    if state == 'Failed' or state == 'CreateFailed':
                        module_logger.error('Provisioning errors: {}'.format(self.provisioning_errors))
            elif status_errors_present:
                if self.status and self.status.errors:
                    errors = self.status.errors
                else:
                    errors = self.provisioning_errors
                if show_output:
                    module_logger.error('There were errors reported from ComputeInstance:\n{}'.format(errors))

    def _terminal_state_reached(self):
        """Terminal state reached.

        :return: Indicates whether the terminal state reached.
        :rtype: bool
        """
        state = self._get_state()
        if state == 'CreateFailed' or state == 'Canceled' or state == 'Running' or state == 'Ready':
            return True
        return False

    def _status_errors_present(self):
        """Return status error.

        :return: Indicates whether the errors present.
        :rtype: bool
        """
        if (self.status and self.status.errors) or self.provisioning_errors:
            return True
        return False

    def _get_state(self):
        """Return current state.

        :return: Current state of the compute instance.
        :rtype: str
        """
        if self.status and self.status.state:
            return self.status.state.capitalize()
        return self.provisioning_state.capitalize()

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily useful for manual polling of compute state.
        """
        instance = ComputeInstance(self.workspace, self.name)
        self.modified_on = instance.modified_on
        self.provisioning_state = instance.provisioning_state
        self.provisioning_errors = instance.provisioning_errors
        self.instance_resource_id = instance.cluster_resource_id
        self.instance_location = instance.cluster_location
        self.vm_size = instance.vm_size
        self.ssh_public_access = instance.ssh_public_access
        self.admin_username = instance.admin_username
        self.admin_user_ssh_public_key = instance.admin_user_ssh_public_key
        self.ssh_port = instance.ssh_port
        self.status = instance.status
        self.public_ip_address = instance.public_ip_address
        self.private_ip_address = instance.private_ip_address
        self.applications = instance.applications
        self.errors = instance.errors
        self.assigned_user_object_id = instance.assigned_user_object_id
        self.assigned_user_tenant_id = instance.assigned_user_tenant_id

    def get_status(self):
        """Retrieve the current detailed status for the ComputeInstance.

        :return: A detailed status object for the compute
        :rtype: azureml.core.compute.computeinstance.ComputeInstanceStatus
        """
        self.refresh_state()
        if not self.status:
            state = self.provisioning_state.capitalize()
            if state == 'Creating':
                module_logger.info('ComputeInstance is getting created. Consider calling wait_for_completion() first')
            elif state == 'Failed':
                module_logger.error('ComputeInstance is in a failed state, try deleting and recreating')
            else:
                module_logger.info('Current provisioning state of ComputeInstance is "{}"'.format(state))
            return None

        return self.status

    def delete(self, wait_for_completion=False, show_output=False):
        """Remove the ComputeInstance object from its associated workspace.

        :param wait_for_completion: Whether to wait for the deletion. Defaults to False.
        :type wait_for_completion: bool
        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool

        .. remarks::

            If this object was created through Azure ML,
            the corresponding cloud based objects will also be deleted. If this object was created externally and only
            attached to the workspace, it will raise exception and nothing will be changed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('delete')
        if wait_for_completion:
            self._wait_for_deletion(show_output=show_output)

    def detach(self):
        """Detach is not supported for ComputeInstance object. Use :meth:`delete` instead.

        :raises azureml.exceptions.ComputeTargetException: The operation is not supproted.
        """
        raise ComputeTargetException('Detach is not supported for ComputeInstance object. Try to use delete instead.')

    def serialize(self):
        """Convert this ComputeInstance object into a JSON serialized dictionary.

        :return: The JSON representation of this ComputeInstance object.
        :rtype: dict
        """
        subnet_id = None
        personal_ci_settings = None
        if self.vnet_resourcegroup_name and self.vnet_name and self.subnet_name:
            subnet_id = "/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks" \
                        "/{2}/subnets/{3}".format(self.workspace.subscription_id, self.vnet_resourcegroup_name,
                                                  self.vnet_name, self.subnet_name)

        if self.assigned_user_object_id and self.assigned_user_tenant_id:
            personal_ci_settings = {'assignedUser': {'objectId': self.assigned_user_object_id,
                                                     'tenantId': self.assigned_user_tenant_id}}
        instance_properties = {'vmSize': self.vm_size,
                               'applications': self.applications,
                               'connectivityEndpoints': {'publicIpAddress': self.public_ip_address,
                                                         'privateIpAddress': self.private_ip_address},
                               'sshSettings': {'sshPublicAccess': "Enabled" if self.ssh_public_access else "Disabled",
                                               'adminUserName': self.admin_username,
                                               'adminPublicKey': self.admin_user_ssh_public_key,
                                               'sshPort': self.ssh_port},
                               'personalComputeInstanceSettings': personal_ci_settings,
                               'subnet': {'id': subnet_id},
                               'errors': self.errors}
        instance_status = self.status.serialize() if self.status else None
        instance_properties = {'description': self.description,
                               'computeType': self.type,
                               'computeLocation': self.location,
                               'resourceId': self.cluster_resource_id,
                               'provisioningErrors': self.provisioning_errors,
                               'provisioningState': self.provisioning_state,
                               'properties': instance_properties,
                               'status': instance_status}
        return {'id': self.id, 'name': self.name, 'location': self.location, 'tags': self.tags,
                'properties': instance_properties}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into a ComputeInstance object.

        This fails if the provided workspace is not the workspace the ComputeInstance is associated with.

        :param workspace: The workspace object the ComputeInstance object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a ComputeInstance object.
        :type object_dict: dict
        :return: The ComputeInstance representation of the provided JSON object.
        :rtype: azureml.core.compute.computeinstance.ComputeInstance
        :raises azureml.exceptions.ComputeTargetException:
        """
        ComputeInstance._validate_get_payload(object_dict)
        target = ComputeInstance(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != ComputeInstance._compute_type:
            raise ComputeTargetException('Invalid payload, not "{}":\n'
                                         '{}'.format(ComputeInstance._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))
        if payload['properties']['properties']:
            for instance_key in ['vmSize', 'sshSettings',
                                 'createdBy', 'errors', 'state']:
                if instance_key not in payload['properties']['properties']:
                    raise ComputeTargetException('Invalid payload, missing '
                                                 '["properties"]["properties"]["{}"]:\n'
                                                 '{}'.format(instance_key, payload))

    def get(self):
        """Return ComputeInstance object.

        :return: The ComputeInstance representation of the provided JSON object.
        :rtype: azureml.core.compute.computeinstance.ComputeInstance
        :raises azureml.exceptions.ComputeTargetException:
        """
        return ComputeTarget._get(self.workspace, self.name)

    def start(self, wait_for_completion=False, show_output=False):
        """Start the ComputeInstance.

        :param wait_for_completion: Whether to wait for the state update. Defaults to False.
        :type wait_for_completion: bool
        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool
        :return: None
        :rtype: None
        :raises azureml.exceptions.ComputeTargetException:
        """
        self._update_instance_state("start", wait_for_completion, show_output)

    def stop(self, wait_for_completion=False, show_output=False):
        """Stop the ComputeInstance.

        :param wait_for_completion: Whether to wait for the state update. Defaults to False.
        :type wait_for_completion: bool
        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool
        :return: None
        :rtype: None
        :raises azureml.exceptions.ComputeTargetException:
        """
        self._update_instance_state("stop", wait_for_completion, show_output)

    def restart(self, wait_for_completion=False, show_output=False):
        """Restart the ComputeInstance.

        :param wait_for_completion: Boolean to wait for the state update. Defaults to False.
        :type wait_for_completion: bool
        :param show_output: Boolean to provide more verbose output. Defaults to False.
        :type show_output: bool
        :return: None
        :rtype: None
        :raises: azureml.exceptions.ComputeTargetException:
        """
        self._update_instance_state("restart", wait_for_completion, show_output)

    def _update_instance_state(self, state, wait_for_completion=False, show_output=False):
        """Update the ComputeInstance state.

        :param state: The state of the ComputeInstance. Supported states are 'start', 'stop', and 'restart'.
        :type state: str
        :param wait_for_completion: Whether to wait for the state update. Defaults to False.
        :type wait_for_completion: bool
        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool
        :return: None
        :rtype: None
        :raises azureml.exceptions.ComputeTargetException:
        """
        retry_interval = 5
        endpoint = self._get_compute_endpoint(self.workspace, self.name) + '/' + state
        headers = self.workspace._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}

        self.refresh_state()
        current_state = self.status.state.lower()

        rasie_error = current_state == "createfailed" or \
            (current_state == "stopped" and state in ["restart", "stop"]) or \
            (current_state.endswith("running") and state == "start")

        if rasie_error:
            raise ComputeTargetException("ComputeInstance current state ({}) does not support {} operation.\n"
                                         .format(self.status.state, state.capitalize()))

        while True:
            resp = ClientBase._execute_func(get_requests_session().post, endpoint, params=params, headers=headers)
            try:
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError:
                if "There is already an active operation submitted" in resp.text and wait_for_completion:
                    time.sleep(retry_interval)
                    continue
                raise ComputeTargetException('Received bad response from Resource Provider:\n'
                                             'Response Code: {}\n'
                                             'Headers: {}\n'
                                             'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        if wait_for_completion:
            self._wait_for_status_update(state, retry_interval, show_output)
            # Wait for MLC long running operation to complete
            if 'Azure-AsyncOperation' in resp.headers:
                self._operation_endpoint = resp.headers['Azure-AsyncOperation']
                self._wait_for_completion(False)

    def _wait_for_status_update(self, operation, refresh_interval=5, show_output=False):
        """Wait for the ComputeInstance status update.

        :param operation: The operation to be performed on compute instance.
        :type operation: str
        :param refresh_interval: The status refresh interval in seconds.
        :type refresh_interval: int
        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool
        :return: None
        :rtype: None
        :raises azureml.exceptions.ComputeTargetException:
        """
        prev_state = None
        current_state = None
        expected_state = "Stopped" if operation == "stop" else "Running"

        while True:
            time.sleep(refresh_interval)
            self.refresh_state()
            current_state = self._get_state()

            if show_output:
                if current_state != prev_state:
                    if prev_state is None:
                        sys.stdout.write('{}'.format(current_state))
                    else:
                        sys.stdout.write('\n{}'.format(current_state))
                elif current_state:
                    sys.stdout.write('.')
                sys.stdout.flush()

            terminal_state_reached = (current_state.endswith(expected_state) or current_state.endswith("Failed"))

            if terminal_state_reached:
                break

            prev_state = current_state

        if current_state.endswith("Failed"):
            raise ComputeTargetException(
                '{operation} operation failed. Errors: {errors}'
                .format(operation=operation.capitalize(), errors=self.status.errors))

        if show_output:
            sys.stdout.write('\n')
            sys.stdout.flush()

            if current_state.endswith(expected_state):
                module_logger.info("{operation} operation completed successfully.".format(
                    operation=operation.capitalize()
                ))

    def _wait_for_deletion(self, refresh_interval=5, show_output=False):
        """Wait for the ComputeInstance delete completion.

        :param refresh_interval: The status refresh interval in seconds.
        :type refresh_interval: int
        :param show_output: Whether to provide more verbose output. Defaults to False.
        :type show_output: bool
        :return: None
        :rtype: None
        :raises azureml.exceptions.ComputeTargetException:
        """
        prev_state = None
        current_state = None
        deleted_state = 'Deleted'

        while True:
            try:
                self.refresh_state()
                current_state = self.provisioning_state
            except ComputeTargetException as e:
                if 'ComputeTargetNotFound' in e.message:
                    current_state = deleted_state
                else:
                    raise e

            terminal_state_reached = (current_state == deleted_state or current_state.endswith("Failed"))

            if show_output:
                if current_state != prev_state:
                    sys.stdout.write('{}{}'.format('' if prev_state is None else '\n', current_state))
                elif current_state:
                    sys.stdout.write('.')
                if terminal_state_reached:
                    sys.stdout.write('\n')
                if current_state == deleted_state:
                    module_logger.info('Compute instance has successfully deleted.')
                sys.stdout.flush()

            if terminal_state_reached:
                break

            prev_state = current_state
            time.sleep(refresh_interval)

        if current_state.endswith("Failed"):
            raise ComputeTargetException(
                'Delete compute operation failed. Errors: {errors}'.format(errors=self.status.errors))

    def get_active_runs(self, type=None, tags=None, properties=None, status=None):
        """Return a generator of the runs for this compute.

        :param type: Filter the returned generator of runs by the provided type. See
            :func:`azureml.core.Run.add_type_provider` for creating run types.
        :type type: str
        :param tags: Filter runs by "tag" or {"tag": "value"}
        :type tags: str or dict
        :param properties: Filter runs by "property" or {"property": "value"}
        :type properties: str or dict
        :param status: Run status, can be "Running" or "Queued".
        :type status: str
        :return: A generator of azureml._restclient.models.RunDto
        :rtype: builtin.generator
        """
        workspace_client = WorkspaceClient(self.workspace.service_context)
        return workspace_client.get_runs_by_compute(
            compute_name=self.name,
            type=type,
            tags=tags,
            properties=properties,
            status=status)

    @staticmethod
    def supported_vmsizes(workspace, location=None):
        """List the supported VM sizes in a region.

        :param workspace: The workspace.
        :type workspace: azureml.core.Workspace
        :param location: The location of the instance. If not specified, the default is the workspace location.
        :type location: str
        :return: A list of supported VM sizes in a region with name of the VM, VCPUs, and RAM.
        :rtype: builtin.list
        """
        paginated_results = []
        if not workspace:
            return paginated_results

        if not location:
            location = workspace.location

        vm_size_fmt = '{}/subscriptions/{}/providers/Microsoft.MachineLearningServices/locations/{}/vmSizes'
        resource_manager_endpoint = ComputeTarget._get_resource_manager_endpoint(workspace)
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
    def _get_user_display_name(workspace, user_id, tenant_id):
        """Get the display name of the user.

        :param user_id: The user id.
        :type user_id: str
        :param tenant_id: The user tenant id.
        :type tenant_id: str
        :return: Display name of the given user id, or None if it cannot fetch the user name using the graph token.
        :rtype: str or None
        """
        if not workspace or not user_id or not tenant_id:
            return None
        try:
            key = user_id + tenant_id

            if key in ComputeInstance._user_display_info:
                return ComputeInstance._user_display_info[key]

            endpoint = '{graph_resource_id}/{tenant_id}/users/{user_id}'.format(
                graph_resource_id=workspace._auth._cloud_type.endpoints.active_directory_graph_resource_id,
                tenant_id=tenant_id,
                user_id=user_id)
            headers = {"Authorization": "Bearer " + workspace._auth._get_graph_token()}
            params = {'api-version': GRAPH_API_VERSION}
            resp = ClientBase._execute_func(get_requests_session().get, endpoint, params=params, headers=headers)
            resp.raise_for_status()
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            result = json.loads(content)
            display_name = result['displayName']
            ComputeInstance._user_display_info[key] = display_name
            return display_name
        except AuthenticationException:
            module_logger.warning("Fetching the graph token failed. Proceed with no display name.")
            ComputeInstance._user_display_info[key] = None
            return None
        except Exception:
            # Set the display name to None if the user is unauthorized to call graph api
            if resp.status_code in [401, 403]:
                ComputeInstance._user_display_info[key] = None
            return None


class ComputeInstanceProvisioningConfiguration(ComputeTargetProvisioningConfiguration):
    """Represents configuration parameters for provisioning ComputeInstance targets.

    Use the :meth:`provisioning_configuration
    <azureml.core.compute.computeinstance.ComputeInstance.provisioning_configuration>` method of the
    ComputeInstance class to create a ComputeInstanceProvisioningConfiguration object.

    :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
        Note that not all sizes are available in all regions, as
        detailed in the previous link. Defaults to Standard_DS3_V2.
    :type vm_size: str
    :param ssh_public_access: Indicates the state of the public SSH port. Possible values are:
        * False - The public ssh port is closed.
        * True - The public ssh port is open.
    :type ssh_public_access: bool
    :param admin_user_ssh_public_key: The SSH public key of the administrator user account.
    :type admin_user_ssh_public_key: str
    :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
    :type vnet_resourcegroup_name: str
    :param vnet_name: The name of the virtual network.
    :type vnet_name: str
    :param subnet_name: The name of the subnet inside the VNet.
    :type subnet_name: str
    :param tags: An optional dictionary of key value tags to associate with the ComputeInstance object.
    :type tags: dict[str, str]
    :param description: An optional description for the ComputeInstance object.
    :type description: str
    """

    def __init__(self, vm_size='', ssh_public_access=False, admin_user_ssh_public_key=None,
                 vnet_resourcegroup_name=None, vnet_name=None, subnet_name=None, tags=None, description=None,
                 assigned_user_object_id=None, assigned_user_tenant_id=None):
        """Create a configuration object for provisioning a ComputeInstance target.

        :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
            Note that not all sizes are available in all regions, as
            detailed in the previous link. Defaults to Standard_DS3_V2.
        :type vm_size: str
        :param ssh_public_access: Indicates the state of the public SSH port. Possible values are:
            * False - The public ssh port is closed.
            * True - The public ssh port is open.
        :type ssh_public_access: bool
        :param admin_user_ssh_public_key: The SSH public key of the administrator user account.
        :type admin_user_ssh_public_key: str
        :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
        :type vnet_resourcegroup_name: str
        :param vnet_name: The name of the virtual network.
        :type vnet_name: str
        :param subnet_name: The name of the subnet inside the vnet.
        :type subnet_name: str
        :param tags: An optional dictionary of key value tags to associate with the ComputeInstance object.
        :type tags: dict[str, str]
        :param description: An optional description for the ComputeInstance object.
        :type description: str
        :param assigned_user_object_id: The AAD Object ID of the assigned user (preview).
        :type assigned_user_object_id: str
        :param assigned_user_tenant_id: The AAD Tenant ID of the assigned user (preview).
        :type assigned_user_tenant_id: str
        :return: A configuration object to be used when creating a ComputeInstance object.
        :rtype: ComputeInstanceProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        if not vm_size:
            vm_size = 'Standard_DS3_V2'

        super(ComputeInstanceProvisioningConfiguration, self).__init__(ComputeInstance, None)
        self.vm_size = vm_size
        self.ssh_public_access = ssh_public_access
        self.admin_user_ssh_public_key = admin_user_ssh_public_key
        self.vnet_resourcegroup_name = vnet_resourcegroup_name
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name
        self.tags = tags
        self.description = description
        self.assigned_user_object_id = assigned_user_object_id
        self.assigned_user_tenant_id = assigned_user_tenant_id
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        if any([self.vnet_name, self.vnet_resourcegroup_name, self.subnet_name]) and \
                not all([self.vnet_name, self.vnet_resourcegroup_name, self.subnet_name]):
            raise ComputeTargetException('Invalid configuration, not all virtual net information provided. '
                                         'To use a custom virtual net, please provide vnet name, vnet resource '
                                         'group and subnet name')


class ComputeInstanceStatus(object):
    """Represents detailed status information about a ComputeInstance object.

    Use the :meth:`azureml.core.compute.computeinstance.ComputeInstance.get_status` method of the
    :class:`azureml.core.compute.computeinstance.ComputeInstance` class to return status information.

    .. remarks::

        Initialize a ComputeInstanceStatus object

    :param creation_time: The instance creation time.
    :type creation_time: datetime.datetime
    :param created_by_user_name: Information on the user who created this ComputeInstance compute.
    :type created_by_user_name: str
    :param created_by_user_id: Uniquely identifies a user within an organization.
    :type created_by_user_id: str
    :param created_by_user_org: Uniquely identifies user Azure Active Directory organization.
    :type created_by_user_org: str
    :param errors: A list of error details, if any exist.
    :type errors: builtin.list
    :param modified_time: The instance modification time.
    :type modified_time: datetime.datetime
    :param state: The current state of this ComputeInstance object.
    :type state: str
    :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
        Note that not all sizes are available in all regions, as
        detailed in the previous link.
    :type vm_size: str
    """

    def __init__(self, creation_time, created_by_user_name, created_by_user_id, created_by_user_org,
                 errors, modified_time, state, vm_size):
        """Initialize a ComputeInstanceStatus object.

        :param creation_time: The instance creation time.
        :type creation_time: datetime.datetime
        :param created_by_user_name: Information on the user who created this ComputeInstance compute.
        :type created_by_user_name: str
        :param created_by_user_id: Uniquely identifies a user within an organization.
        :type created_by_user_id: str
        :param created_by_user_org: Uniquely identifies user Azure Active Directory organization.
        :type created_by_user_org: str
        :param errors: A list of error details, if any exist.
        :type errors: builtin.list
        :param modified_time: The instance modification time.
        :type modified_time: datetime.datetime
        :param state: The current state of this ComputeInstance object.
        :type state: str
        :param vm_size: The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.
            Note that not all sizes are available in all regions, as
            detailed in the previous link.
        :type vm_size: str
        """
        self.creation_time = creation_time
        self.created_by_user_name = created_by_user_name
        self.created_by_user_id = created_by_user_id
        self.created_by_user_org = created_by_user_org
        self.errors = errors
        self.modified_time = modified_time
        self.state = state
        self.vm_size = vm_size

    def __repr__(self):
        """Return the string representation of the ComputeInstanceStatus object.

        :return: String representation of the ComputeInstanceStatus object
        :rtype: str
        """
        return json.dumps(self.serialize(), indent=2)

    def serialize(self):
        """Convert this ComputeInstanceStatus object into a JSON serialized dictionary.

        :return: The JSON representation of this ComputeInstanceStatus object.
        :rtype: dict
        """
        creation_time = self.creation_time.isoformat() if self.creation_time else None
        modified_time = self.modified_time.isoformat() if self.modified_time else None
        return {'errors': self.errors,
                'creationTime': creation_time,
                'createdBy': {'userObjectId': self.created_by_user_id,
                              'userTenantId': self.created_by_user_org,
                              'userName': self.created_by_user_name},
                'modifiedTime': modified_time,
                'state': self.state,
                'vmSize': self.vm_size}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into a ComputeInstanceStatus object.

        :param object_dict: A JSON object to convert to a ComputeInstanceStatus object.
        :type object_dict: dict
        :return: The ComputeInstanceStatus representation of the provided JSON object.
        :rtype: azureml.core.compute.computeinstance.ComputeInstanceStatus
        :raises azureml.exceptions.ComputeTargetException:
        """
        if not object_dict:
            return None
        creation_time = parse(object_dict['createdOn']) \
            if 'createdOn' in object_dict else None
        modified_time = parse(object_dict['modifiedOn']) \
            if 'modifiedOn' in object_dict else None
        instance_properties = object_dict['properties'] \
            if 'properties' in object_dict else None
        vm_size = instance_properties['vmSize'] \
            if instance_properties and 'vmSize' in instance_properties else None
        state = instance_properties['state'] \
            if instance_properties and 'state' in instance_properties else None
        errors = instance_properties['errors'] \
            if instance_properties and 'errors' in instance_properties else None
        created_by = instance_properties['createdBy'] \
            if instance_properties and 'createdBy' in instance_properties else None
        created_by_user_name = created_by['userName'] \
            if created_by and 'userName' in created_by else None
        created_by_user_id = created_by['userId'] \
            if created_by and 'userId' in created_by else None
        created_by_user_org = created_by['userOrgId'] \
            if created_by and 'userOrgId' in created_by else None

        return ComputeInstanceStatus(creation_time, created_by_user_name, created_by_user_id, created_by_user_org,
                                     errors, modified_time, state, vm_size)
