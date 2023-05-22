# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Data Science Virtual Machine compute targets in Azure Machine Learning."""

import copy
import json
import requests
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml._compute._util import dsvm_payload_template
from azureml._compute._util import get_requests_session
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetProvisioningConfiguration
from azureml.exceptions import ComputeTargetException
from azureml._restclient.clientbase import ClientBase


class DsvmCompute(ComputeTarget):
    """Manages a Data Science Virtual Machine compute target in Azure Machine Learning.

    The Azure Data Science Virtual Machine (DSVM) is a pre-configured data science and AI development
    environment in Azure. The VM offers a curated choice of tools and frameworks for full-lifecycle
    machine learning development.
    For more information, see `Data Science Virtual
    Machine <https://docs.microsoft.com/azure/machine-learning/how-to-configure-environment#dsvm>`_.

    :param workspace: The workspace object containing the DsvmCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the of the DsvmCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'VirtualMachine'

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
        resource_manager_endpoint = self._get_resource_manager_endpoint(workspace)
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
        is_attached = obj_dict['properties']['isAttachedCompute']
        vm_size = obj_dict['properties']['properties']['virtualMachineSize'] \
            if 'virtualMachineSize' in obj_dict['properties']['properties'] else None
        address = obj_dict['properties']['properties']['address'] \
            if 'address' in obj_dict['properties']['properties'] else None
        ssh_port = obj_dict['properties']['properties']['sshPort'] \
            if 'sshPort' in obj_dict['properties']['properties'] else None

        super(DsvmCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags, description,
                                             created_on, modified_on, provisioning_state, provisioning_errors,
                                             cluster_resource_id, cluster_location, workspace, mlc_endpoint, None,
                                             workspace._auth, is_attached)
        self.vm_size = vm_size
        self.address = address
        self.ssh_port = ssh_port

    def __repr__(self):
        """Return the string representation of the DsvmCompute object.

        :return: String representation of the DsvmCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def _create(workspace, name, provisioning_configuration):  # pragma: no cover
        """DEPRECATED.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param provisioning_configuration:
        :type provisioning_configuration: azureml.core.compute.dsvm.DsvmProvisioningConfiguration
        :return:
        :rtype: azureml.core.compute.dsvm.DsvmCompute
        """
        print("""
        DEPRECATED
        This class will be deprecated soon and we will remove support for it in an upcoming release.
        Please use the \"AmlCompute\" class instead, or spin up a VM in Azure and attach it using RemoteCompute().
        Use !help AmlCompute to learn more.
        """)
        compute_create_payload = DsvmCompute._build_create_payload(provisioning_configuration, workspace.location)
        return ComputeTarget._create_compute_target(workspace, name, compute_create_payload, DsvmCompute)

    @staticmethod
    def provisioning_configuration(vm_size=None, ssh_port=None, location=None):
        """Create a configuration object for provisioning a DsvmCompute target.

        :param vm_size: Specifies the size of the VM to provision. More details can be found here:
            https://aka.ms/azureml-vm-details. Defaults to Standard_DS3_v2.
        :type vm_size: str
        :param ssh_port: The SSH port to open on the VM.
        :type ssh_port: str
        :param location: Location to provision cluster in. If not specified, will default to workspace location.
            Available regions for different desired VM sizes can be found here:
            https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=virtual-machines
        :type location: str
        :return: A configuration object to be used when creating a Compute object.
        :rtype: azureml.core.compute.dsvm.DsvmProvisioningConfiguration
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        config = DsvmProvisioningConfiguration(vm_size, ssh_port, location)
        return config

    @staticmethod
    def _build_create_payload(config, location):  # pragma: no cover
        """DEPRECATED.

        Construct the payload needed to create a DSVM.

        :param config:
        :type config: azureml.core.compute.dsvm.DsvmProvisioningConfiguration
        :param location:
        :type location: str
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(dsvm_payload_template)
        del(json_payload['properties']['resourceId'])
        json_payload['location'] = location
        if config.ssh_port:
            json_payload['properties']['properties']['sshPort'] = config.ssh_port
        if config.vm_size:
            json_payload['properties']['properties']['virtualMachineSize'] = config.vm_size
        else:
            del (json_payload['properties']['properties']['virtualMachineSize'])
        if config.location:
            json_payload['properties']['computeLocation'] = config.location
        else:
            del (json_payload['properties']['computeLocation'])
        return json_payload

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = DsvmCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location
        self.vm_size = cluster.vm_size
        self.address = cluster.address
        self.ssh_port = cluster.ssh_port

    def delete(self):
        """Remove the DsvmCompute object from its associated workspace.

        .. remarks::

            If this object was created through Azure Machine Learning,
            the corresponding cloud based objects will also be deleted. If this object was created externally and only
            attached to the workspace, it will raise exception and nothing will be changed.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        self._delete_or_detach('delete')

    def detach(self):
        """Detach is not supported for a DsvmCompute object. Use :meth:`delete` instead.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        raise ComputeTargetException('Detach is not supported for DSVM object. Try to use delete instead.')

    def get_credentials(self):
        """Retrieve the credentials for the DsvmCompute target.

        :return: The credentials for the DsvmCompute target.
        :rtype: dict
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        endpoint = self._mlc_endpoint + '/listKeys'
        headers = self._auth.get_authentication_header()
        ComputeTarget._add_request_tracking_headers(headers)
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = ClientBase._execute_func(get_requests_session().post, endpoint, params=params, headers=headers)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from MLC:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        creds_content = json.loads(content)
        return creds_content

    def serialize(self):
        """Convert this DsvmCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this DsvmCompute object.
        :rtype: dict
        """
        dsvm_properties = {'vmSize': self.vm_size, 'address': self.address, 'ssh-port': self.ssh_port}
        cluster_properties = {'computeType': self.type, 'computeLocation': self.cluster_location,
                              'description': self.description, 'resourceId': self.cluster_resource_id,
                              'isAttachedCompute': self.is_attached,
                              'provisioningErrors': self.provisioning_errors,
                              'provisioningState': self.provisioning_state, 'properties': dsvm_properties}
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': cluster_properties}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into a DsvmCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the DsvmCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a DsvmCompute object.
        :type object_dict: dict
        :return: The DsvmCompute representation of the provided JSON object.
        :rtype: azureml.core.compute.dsvm.DsvmCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        DsvmCompute._validate_get_payload(object_dict)
        target = DsvmCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != DsvmCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(DsvmCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))


class DsvmProvisioningConfiguration(ComputeTargetProvisioningConfiguration):
    """Represents configuration parameters for provisioning DsvmCompute targets.

    Use the :meth:`azureml.core.compute.DsvmCompute.provisioning_configuration` method of the
    :class:`azureml.core.compute.DsvmCompute` class to specify provisioning configuration.

    :param vm_size: Specifies the size of the VM to provision. More details can be found here:
        https://aka.ms/azureml-vm-details. If not specified, defaults to Standard_DS3_v2.
    :type vm_size: str
    :param ssh_port: The SSH port to open on the VM.
    :type ssh_port: str
    :param location: The location to provision the cluster in. If not specified, will default to workspace location.
        Available regions for different desired VM sizes can be found here:
        https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=virtual-machines
    :type location: str
    """

    def __init__(self, vm_size='', ssh_port=None, location=None):
        """Create a configuration object for provisioning a DSVM target.

        :param vm_size: Specifies the size of the VM to provision. More details can be found here:
            https://aka.ms/azureml-vm-details. If not specified, defaults to Standard_DS3_v2.
        :type vm_size: str
        :param ssh_port: The SSH port to open on the VM.
        :type ssh_port: str
        :param location: The location to provision the cluster in. If not specified, will default to workspace
            location. Available regions for different desired VM sizes can be found here:
            https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=virtual-machines
        :type location: str
        :return: A configuration object to be used when creating a Compute object.
        :rtype: azureml.core.compute.dsvm.DsvmProvisioningConfiguration
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        super(DsvmProvisioningConfiguration, self).__init__(DsvmCompute, location)
        self.ssh_port = ssh_port
        self.vm_size = vm_size
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        pass
