# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Manages Synapse compute targets in Azure Machine Learning service."""

import copy
import re
from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT, MLC_LIST_COMPUTES_FMT
from azureml.core.compute import ComputeTarget
from azureml.core.linked_service import LinkedService
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml._compute._util import synapse_compute_template
from azureml.exceptions import ComputeTargetException


@experimental
class SynapseCompute(ComputeTarget):
    """Manages an Synapse compute target in Azure Machine Learning. Currently it only supports spark.

    Azure Synapse is an integrated analytics service that accelerates time to insight across data
    warehouses and big data analytics systems. At its core, Azure Synapse brings together the best
    of SQL technologies used in enterprise data warehousing, Spark technologies used for big data,
    and Pipelines for data integration and ETL/ELT.
    For more information, see `What is an Synapse spark pool instance?
    <https://docs.microsoft.com/en-us/azure/synapse-analytics/spark/apache-spark-overview>`_.
    """

    _compute_type = 'SynapseSpark'

    def _initialize(self, workspace, obj_dict):
        """Initialize implementation method.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
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
        is_attached = obj_dict['properties']['isAttachedCompute']

        spark_version = (obj_dict['properties']['properties']['sparkVersion']
                         if 'sparkVersion' in obj_dict['properties']['properties'] else None)

        node_size_family = (obj_dict['properties']['properties']['nodeSizeFamily']
                            if 'nodeSizeFamily' in obj_dict['properties']['properties'] else None)

        node_count = (obj_dict['properties']['properties']['nodeCount']
                      if 'nodeCount' in obj_dict['properties']['properties'] else None)

        node_size = (obj_dict['properties']['properties']['nodeSize']
                     if 'nodeSize' in obj_dict['properties']['properties'] else None)

        auto_scale_enabled = (obj_dict['properties']['properties']['autoScaleProperties']['enabled']
                              if 'autoScaleProperties' in obj_dict['properties']['properties']
                              and 'enabled' in obj_dict['properties']['properties']['autoScaleProperties'] else None)

        max_node_count = (obj_dict['properties']['properties']['autoScaleProperties']['maxNodeCount']
                          if 'autoScaleProperties' in obj_dict['properties']['properties']
                          and 'maxNodeCount' in obj_dict['properties']['properties']['autoScaleProperties'] else None)

        min_node_count = (obj_dict['properties']['properties']['autoScaleProperties']['minNodeCount']
                          if 'autoScaleProperties' in obj_dict['properties']['properties']
                          and 'minNodeCount' in obj_dict['properties']['properties']['autoScaleProperties'] else None)

        auto_pause_enabled = (obj_dict['properties']['properties']['autoPauseProperties']['enabled']
                              if 'autoPauseProperties' in obj_dict['properties']['properties']
                              and 'enabled' in obj_dict['properties']['properties']['autoPauseProperties'] else None)

        auto_pause_in_minutes = (obj_dict['properties']['properties']['autoPauseProperties']['delayInMinutes']
                                 if 'autoPauseProperties' in obj_dict['properties']['properties']
                                 and 'delayInMinutes' in obj_dict['properties']['properties']['autoPauseProperties']
                                 else None)

        super(SynapseCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags,
                                                description, created_on, modified_on, provisioning_state,
                                                provisioning_errors, cluster_resource_id, cluster_location,
                                                workspace, mlc_endpoint, None, workspace._auth, is_attached)

        self.spark_version = spark_version
        self.node_count = node_count
        self.node_size_family = node_size_family
        self.node_size = node_size
        self.auto_scale_enabled = auto_scale_enabled
        self.max_node_count = max_node_count
        self.min_node_count = min_node_count
        self.auto_pause_enabled = auto_pause_enabled
        self.auto_pause_in_minutes = auto_pause_in_minutes

    def __repr__(self):
        """Return the string representation of the SynapseCompoute object.

        :return: String representation of the SynapseCompoute object
        :rtype: str
        """
        formatted_info = ',\n'.join(
            ["{}: {}".format(k, v) for k, v in self.serialize().items()])
        return formatted_info

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        Based on the current state of the corresponding cloud object.

        Primarily useful for manual polling of compute state.
        """
        cluster = SynapseCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location

    @staticmethod
    def _build_attach_payload(resource_id):
        """Build attach payload.

        :param resource_id: resource id of the synapse spark pool.
        :type resource_id: str
        :return: payload for attaching compute API call.
        :rtype: dict
        """
        if not resource_id:
            raise ComputeTargetException('Error, missing resource_id.')

        json_payload = copy.deepcopy(synapse_compute_template)
        json_payload['properties']['resourceId'] = resource_id
        return json_payload

    @staticmethod
    def attach_configuration(linked_service, type, pool_name):
        """Create a configuration object for attaching an Synapse compute target.

        :param linked_service: You must link the synapse workspace first
                                 and then you can attach the spark pools within the synapse workspace into AML.
        :type linked_service: azureml.core.LinkedService
        :param type: compute target type, currently only support SynapseSpark.
        :type type: str
        :param pool_name: Name synapse spark pool
        :type pool_name: str
        """
        config = SynapseAttachConfiguration(linked_service, type, pool_name)
        return config

    @staticmethod
    def _attach(workspace, name, config):
        """Associates an already existing Synapse spark pool with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace.
                     Does not have to match with the already given name of the compute resource.
        :type name: str
        :param config: Attach configuration object.
        :type config: azureml.core.compute.synapse.SynapseAttachConfiguration
        :return: A SynapseCompute object representation of the compute object.
        :rtype: azureml.core.compute.Synapse.SynapseCompute
        :raises: azureml.exceptions.ComputeTargetException
        """
        def _valid_compute_name(compute_name):
            message = "compute name must be between 2 and 16 characters long. " \
                "Its first character has to be alphanumeric, and " \
                "valid characters include letters, digits, and the - character."
            if not compute_name:
                raise ComputeTargetException('Compute_name cannot be Empty.')
            if not re.match("^[A-Za-z][A-Za-z0-9-]{0,14}[A-Za-z0-9]$", compute_name):
                raise ComputeTargetException(message)

        _valid_compute_name(name)

        linked_service_resource_id = config.linked_service.linked_service_resource_id
        compute_resource_id = linked_service_resource_id + \
            "/bigDataPools/" + config.pool_name

        attach_payload = SynapseCompute._build_attach_payload(
            compute_resource_id)
        return ComputeTarget._attach(workspace, name, attach_payload, SynapseCompute)

    def delete(self):
        """Delete is not supported for Synapse object. Try to use detach instead.

        :raises: azureml.exceptions.ComputeTargetException
        """
        raise ComputeTargetException(
            'Delete is not supported for Synapse object. Try to use detach instead.')

    def detach(self):
        """Detaches the Synapse object from its associated workspace.

        .. remarks::

            No underlying cloud object will be deleted, the
            association will just be removed.

        :raises: azureml.exceptions.ComputeTargetException
        """
        self._delete_or_detach('detach')

    def serialize(self):
        """Convert this SynapseCompute object into a json serialized dictionary.

        :return: The json representation of this SynapseCompute object
        :rtype: dict
        """
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': {'computeType': self.type, 'computeLocation': self.cluster_location,
                               'description': self.description,
                               'resourceId': self.cluster_resource_id,
                               'provisioningErrors': self.provisioning_errors,
                               'provisioningState': self.provisioning_state,
                               'properties': {'sparkVersion': self.spark_version,
                                              'nodeCount': self.node_count,
                                              'nodeSizeFamily': self.node_size_family,
                                              'nodeSize': self.node_size,
                                              'autoScaleProperties': {'enabled': self.auto_scale_enabled,
                                                                      'maxNodeCount': self.max_node_count,
                                                                      'minNodeCount': self.min_node_count},
                                              'autoPauseProperties': {'enabled': self.auto_pause_enabled,
                                                                      'delayInMinutes': self.auto_pause_in_minutes}
                                              }
                               }
                }

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a json object into a SynapseCompute object.

        Will fail if the provided workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the SynapseCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A json object to convert to a SynapseCompute object.
        :type object_dict: dict
        :return: The SynapseCompute representation of the provided json object.
        :rtype: azureml.core.compute.synapse.SynapseCompute
        :raises: azureml.exceptions.ComputeTargetException
        """
        SynapseCompute._validate_get_payload(object_dict)
        target = SynapseCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != SynapseCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(SynapseCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))


class SynapseAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching SynapseCompute targets.

    Use the ``attach_configuration`` method of the
    :class:`azureml.core.compute.synapse.SynapseCompute` class to
    specify attach parameters.
    """

    def __init__(self, linked_service, type, pool_name):
        """Initialize the configuration object.

        :param linked_service:.
        :type linked_service: azureml.core.LinkedService
        :param type: compute target type, currently only support SynapseSpark
        :type type: str
        :param pool_name: Name of the resource group in which Synapse spark pool is located.
        :type pool_name: str
        """
        super(SynapseAttachConfiguration, self).__init__(SynapseCompute)
        self.linked_service = linked_service
        self.type = type
        self.pool_name = pool_name
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        if not self.linked_service or (not isinstance(self.linked_service, LinkedService)):
            raise ComputeTargetException(
                'A valid linked_service object is required to attach compute.')

        if self.type != "SynapseSpark":
            raise ComputeTargetException(
                'Only SynapseSpark type is supported now.')

        if not self.pool_name:
            raise ComputeTargetException(
                'pool_name must be provided to attach synapse compute')
