# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Azure Machine Learning compute targets in Azure Machine Learning."""

import copy
import re

from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml.exceptions import ComputeTargetException
from azureml._compute._util import to_json_dict
from azureml._compute._util import kubernetes_compute_template
from azureml._base_sdk_common._docstring_wrapper import experimental
from dateutil.parser import parse


@experimental
class KubernetesCompute(ComputeTarget):
    """KubernetesCompute (Preview) is a customer managed K8s cluster attached to a workspace by cluster admin.

    User granted access and quota to the compute can easily specify and submit a one-node or
    distributed multi-node ML workload to the compute. The compute executes in a containerized environment and
    packages your model dependencies in a docker container.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`

    .. remarks::

        In the following example, a persistent compute target provisioned by
        :class:`azureml.contrib.core.compute.KubernetesCompute.KubernetesCompute`
        is created. The ``provisioning_configuration``
        parameter in this example is of type
        :class:`azureml.contrib.core.compute.KubernetesCompute.KubernetesComputeAttachConfiguration`,
        which is a child class of
        :class:`azureml.contrib.core.compute.KubernetesCompute.ComputeTargetAttachConfiguration`.

    :param workspace: The workspace object containing the KubernetesCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the of the KubernetesCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'Kubernetes'

    @staticmethod
    def _attach(workspace, name, config):
        """Associate an existing kube like compute resource with the provided workspace.

        :param workspace: The workspace object the KubernetesCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached..
        :type name: str
        :param config: Attach configuration object.
        :type config: azureml.contrib.core.compute.KubernetesCompute.KubernetesComputeAttachConfiguration
        :return: A KubernetesCompute object representation of the compute object.
        :rtype: azureml.contrib.core.compute.KubernetesCompute.KubernetesCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        attach_payload = KubernetesCompute._build_attach_payload(config, workspace)
        return ComputeTarget._attach(workspace, name, attach_payload, KubernetesCompute)

    @staticmethod
    def attach_configuration(resource_id=None, namespace=None,
                             identity_type=None, identity_ids=None):
        """Create a configuration object for attaching an compute target.

        :param resource_id: The resource id.
        :type resource_id: str
        :param namespace: The Kubernetes namespace to be used by workloads submitted to the compute target.
        :type namespace: str
        :param identity_type: identity type.
        :type identity_type: string
        :param identity_ids: List of resource ids for the user assigned identity. eg.
            ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<id>']
        :type identity_ids: builtin.list[str]
        :return: A configuration object to be used when attaching a KubernetesCompute object.
        :rtype: azureml.contrib.core.compute.KubernetesCompute.KubernetesComputeAttachConfiguration
        """
        config = KubernetesComputeAttachConfiguration(resource_id, namespace,
                                                      identity_type, identity_ids)
        return config

    @staticmethod
    def _build_attach_payload(config, workspace):
        """Build attach payload.

        :param config: the compute configuration.
        :type config: KubeComputeAttachConfiguration
        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(kubernetes_compute_template)
        del (json_payload['properties']['computeLocation'])

        if not config:
            raise ComputeTargetException('Error, missing config.')

        attach_resource_id = config.resource_id
        if not attach_resource_id:
            raise ComputeTargetException(
                'Error, missing resource_id.')

        json_payload['properties'].update(config.to_dict())
        json_payload['properties']['resourceId'] = attach_resource_id
        json_payload['properties']['properties']['namespace'] = config.namespace
        if config.identity_type:
            json_payload['identity']['type'] = config.identity_type
            if config.identity_ids:
                for id in config.identity_ids:
                    json_payload['identity']['userAssignedIdentities'][id] = dict()
            else:
                del(json_payload['identity']['userAssignedIdentities'])
        else:
            del(json_payload['identity'])

        return json_payload

    def wait_for_completion(self, show_output=False, is_delete_operation=False):
        """Wait for the KubernetesCompute cluster to finish provisioning.

        :param show_output: Boolean to provide more verbose output.
        :type show_output: bool
        :param is_delete_operation: Indicates whether the operation is meant for deleting.
        :type is_delete_operation: bool
        :raises azureml.exceptions.ComputeTargetException:
        """
        # For all LROs poll MLC's operation status endpoint first
        if self._operation_endpoint:
            super(KubernetesCompute, self).wait_for_completion(show_output=show_output,
                                                               is_delete_operation=is_delete_operation)
        else:
            if show_output:
                print('Warning: wait_for_completion called after attachment process finished.\n')
                state = self.provisioning_state.capitalize()
                print('Final state of "{}" has been reached\n'.format(state))
                if state == 'Failed':
                    print('Provisioning errors: {}\n'.format(self.provisioning_errors))

    def delete(self):
        """Delete is not supported for an KubernetesCompute object. Use :meth:`detach` instead.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        raise ComputeTargetException(
            'Delete is not supported for KubernetesCompute object. Try to use detach instead.')

    def detach(self):
        """Detach the KubernetesCompute object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        self._delete_or_detach('detach')

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        compute_target = KubernetesCompute(
            self.workspace, self.name)
        self.modified_on = compute_target.modified_on
        self.provisioning_state = compute_target.provisioning_state
        self.provisioning_errors = compute_target.provisioning_errors
        self.cluster_resource_id = compute_target.cluster_resource_id
        self.cluster_location = compute_target.cluster_location
        self.status = compute_target.status
        self.namespace = compute_target.namespace
        self.default_instance_type = compute_target.default_instance_type
        self.instance_types = compute_target.instance_types
        self.identity_type = compute_target.identity_type
        self.identity_ids = compute_target.identity_ids

    def get_status(self):
        """Retrieve the current detailed status for the KubernetesCompute cluster.

        :return: A detailed status object for the cluster
        :rtype: azureml.core.compute.kubernetescompute.KubernetesComputeStatus
        """
        self.refresh_state()
        if not self.status:
            state = self.provisioning_state.capitalize()
            return state
        return self.status

    def get(self):
        """Send GET compute object request to mlc."""
        return ComputeTarget._get(self.workspace, self.name)

    def serialize(self):
        """Convert this KubernetesCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this KubernetesCompute object.
        :rtype: dict
        """
        kubernetescompute_status = self.status.serialize() if self.status else None
        cluster_properties = {'computeType': self.type, 'computeLocation': self.cluster_location,
                              'description': self.description, 'resourceId': self.cluster_resource_id,
                              'provisioningErrors': self.provisioning_errors,
                              'provisioningState': self.provisioning_state,
                              'properties': {'namespace': self.namespace,
                                             'defaultInstanceType': self.default_instance_type,
                                             'instanceTypes': self.instance_types},
                              'status': kubernetescompute_status}
        self.identity = self.get().get('identity', None) if self.identity_type else None
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'identity': self.identity, 'properties': cluster_properties}

    def _initialize(self, workspace, obj_dict):
        """Initialize implementation method.

        :param workspace: The workspace object the KubernetesCompute object is to be associated with.
        :type workspace: azureml.core.Workspace
        :param obj_dict: dictionary containing KubernetesCompute properties such as name.
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        name = obj_dict['name']
        compute_resource_id = MLC_COMPUTE_RESOURCE_ID_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                                 workspace.name, name)
        resource_manager_endpoint = self._get_resource_manager_endpoint(
            workspace)
        mlc_endpoint = '{}{}'.format(
            resource_manager_endpoint, compute_resource_id)
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
        if provisioning_state.lower() == 'failed':
            print('This compute target is in a Failed state.')
            print(f'Current provisioning error: {provisioning_errors}')
        is_attached = obj_dict['properties']['isAttachedCompute'] \
            if 'isAttachedCompute' in obj_dict['properties'] else None
        status = KubernetesComputeStatus.deserialize(obj_dict['properties']) \
            if provisioning_state in ["Succeeded", "Updating"] else None
        self.status = status
        self.namespace = obj_dict['properties']['properties'].get(
            'namespace')
        self.default_instance_type = obj_dict['properties']['properties'].get(
            'defaultInstanceType')
        self.instance_types = obj_dict['properties']['properties'].get(
            'instanceTypes')
        self.public_ip = obj_dict['properties']['properties'].get('publicIp')
        self.vm_size = 'NA'
        self.identity_type = obj_dict['identity']['type'] \
            if 'identity' in obj_dict else None
        self.identity_ids = obj_dict['identity']['userAssignedIdentities'] \
            if self.identity_type and 'userAssignedIdentities' in obj_dict['identity'] else None
        super()._initialize(
            compute_resource_id, name, location, compute_type, tags, description,
            created_on, modified_on, provisioning_state, provisioning_errors,
            cluster_resource_id, cluster_location, workspace, mlc_endpoint, None,
            workspace._auth, is_attached)

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into a KubernetesCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the KubernetesCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a KubernetesCompute object.
        :type object_dict: dict
        :return: The KubernetesCompute representation of the provided JSON object.
        :rtype: azureml.contrib.core.compute.KubernetesCompute.KubernetesCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        KubernetesCompute._validate_get_payload(object_dict)
        target = KubernetesCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    def __repr__(self):
        """Print extra identity information."""
        if self.identity_type:
            self.identity = self.get()['identity']
        return "{}(workspace={}, name={}, id={}, type={}, provisioning_state={}, location={}, identity={}, " \
               "tags={})".format(self.__class__.__name__,
                                 self.workspace.__repr__() if hasattr(self, 'workspace') else None,
                                 self.name if hasattr(self, 'name') else None,
                                 self.id if hasattr(self, 'id') else None,
                                 self.type if hasattr(self, 'type') else None,
                                 self.provisioning_state if hasattr(self, 'provisioning_state') else None,
                                 self.location if hasattr(self, 'location') else None,
                                 self.identity if hasattr(self, 'identity') else None,
                                 self.tags if hasattr(self, 'tags') else None)

    @staticmethod
    def _validate_get_payload(payload):
        """Validate provided payload.

        :param payload: Compute payload.
        :type dict
        :return:
        :rtype: None
        """
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != KubernetesCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(KubernetesCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))

    @staticmethod
    def _validate_identity(identity_type, identity_ids):
        """Validate identity_type and identity_ids fields.

        :param identity_type: Possible values are:
            * SystemAssigned - System assigned identity
            * UserAssigned - User assigned identity. Requires identity id to be set.
        :type identity_type: string
        :param identity_ids: List of resource ids for the user assigned identity.
            eg. ['/subscriptions/<subid>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity
            /userAssignedIdentities/<id>']
        :type identity_ids: builtin.list[str]
        """
        if not identity_type and identity_ids:
            raise ComputeTargetException('Invalid configuration, identity_ids cannot be set when '
                                         'identity type is not user assigned.')
        if identity_type not in ["UserAssigned", "SystemAssigned"]:
            raise ComputeTargetException("Invalid configuration, identity_type must be set to "
                                         "'UserAssigned' or 'SystemAssigned'.")
        elif identity_type == "UserAssigned" and (type(identity_ids) is not list or not identity_ids):
            raise ComputeTargetException("Invalid configuration, please pass the identity_ids of the user assigned "
                                         "identities as a list (eg. ['resource_ID1','resourceID2',...])")
        elif identity_type != "UserAssigned" and identity_ids:
            raise ComputeTargetException('Invalid configuration, identity_ids cannot be set when '
                                         'identity type is not user assigned.')


class KubernetesComputeAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching arc compute targets.

    Use the :meth:`azureml.core.compute.kubernetescompute.KubernetesCompute.attach_configuration` method of the
    :class:`azureml.core.compute.kubernetescompute.KubernetesCompute` class to
    specify attach parameters.
    """

    def __init__(self, resource_id=None, namespace=None, identity_type=None, identity_ids=None):
        """Initialize the configuration object."""
        super(KubernetesComputeAttachConfiguration, self).__init__(KubernetesCompute)
        self.resource_id = resource_id
        self.namespace = namespace
        self.identity_type = identity_type
        self.identity_ids = identity_ids
        self.validate_configuration()

    def to_dict(self):
        """Return to_json_dict function from utils."""
        return to_json_dict(self)

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        if self.resource_id:
            # resource_id is provided, validate resource_id
            aks_arm_type = 'Microsoft.ContainerService/managedClusters'
            arc_arm_type = 'Microsoft.Kubernetes/connectedClusters'
            CLUSTER_TYPE_REGEX = '(?:{}|{})'.format(aks_arm_type, arc_arm_type)
            arm_template = ('/subscriptions/{part}/resourceGroups/{part}'
                            '/providers/{cluster_type}/{part}')
            resource_id_pattern = \
                '(?i)' + arm_template.format(part=r'[\w\-_\.]+', cluster_type=CLUSTER_TYPE_REGEX)
            if not re.match(resource_id_pattern, self.resource_id):
                raise ComputeTargetException('Invalid resource_id provided: {} \n Does not match\n'
                                             'AKS template: {}\n or\n ARC template: {}'
                                             .format(self.resource_id,
                                                     arm_template.format(part='<>', cluster_type=aks_arm_type),
                                                     arm_template.format(part='<>', cluster_type=arc_arm_type)))
        else:
            raise ComputeTargetException('Missing argument: resource_id.')
        if self.identity_type or self.identity_ids:
            KubernetesCompute._validate_identity(self.identity_type, self.identity_ids)


class KubernetesComputeStatus(object):
    """Represents detailed status information about an KubernetesCompute target.

    Use the :meth:`azureml.core.compute.KubernetesCompute.get_status` method of
    the :class:`azureml.core.compute.KubernetesCompute` class to return status information.

    :param creation_time: The cluster creation time.
    :type creation_time: datetime.datetime
    :param modified_time: The cluster modification time.
    :type modified_time: datetime.datetime
    :param provisioning_state: The current provisioning state of the cluster.
    :type provisioning_state: str
    """

    def __init__(self, creation_time, modified_time, provisioning_state):
        """Initialize a KubernetesComputeStatus object.

        :param creation_time: The cluster creation time.
        :type creation_time: datetime.datetime
        :param modified_time: The cluster modification time.
        :type modified_time: datetime.datetime
        :param provisioning_state: The current provisioning state of the cluster.
        :type provisioning_state: str
        """
        self.creation_time = creation_time
        self.modified_time = modified_time
        self.provisioning_state = provisioning_state

    def serialize(self):
        """Convert this KubernetesComputeStatus object into a JSON serialized dictionary.

        :return: The JSON representation of this KubernetesComputeStatus object.
        :rtype: dict
        """
        creation_time = self.creation_time.isoformat() if self.creation_time else None
        modified_time = self.modified_time.isoformat() if self.modified_time else None
        return {'creationTime': creation_time, 'modifiedTime': modified_time,
                'provisioningState': self.provisioning_state}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into an KubernetesComputeStatus object.

        :param object_dict: A JSON object to convert to an KubernetesComputeStatus object.
        :type object_dict: dict
        :return: The KubernetesComputeStatus representation of the provided JSON object.
        :rtype: azureml.core.compute.KubernetesCompute.KubernetesComputeStatus
        :raises azureml.exceptions.ComputeTargetException:
        """
        if not object_dict:
            return None
        creation_time = parse(object_dict['createdOn']) \
            if 'createdOn' in object_dict else None
        modified_time = parse(object_dict['modifiedOn']) \
            if 'modifiedOn' in object_dict else None
        provisioning_state = object_dict['provisioningState'] \
            if 'provisioningState' in object_dict else None
        return KubernetesComputeStatus(creation_time, modified_time, provisioning_state)
