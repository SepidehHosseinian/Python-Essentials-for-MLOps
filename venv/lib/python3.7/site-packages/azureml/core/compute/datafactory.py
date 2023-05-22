# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Datafactory compute targets in Azure Machine Learning."""

import copy
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._util import datafactory_payload_template
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetProvisioningConfiguration
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml.exceptions import ComputeTargetException


class DataFactoryCompute(ComputeTarget):
    """Manages a DataFactory compute target in Azure Machine Learning.

    Azure Data Factory is Azure's cloud ETL service for scale-out serverless data integration and
    data transformation. For more information, see `Azure Data
    Factory <https://docs.microsoft.com/en-us/azure/data-factory/>`_.

    :param workspace: The workspace object containing the DataFactoryCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the DataFactoryCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'DataFactory'

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
            if 'computeLocation' in obj_dict['properties'] else location
        provisioning_state = obj_dict['properties']['provisioningState']
        provisioning_errors = obj_dict['properties']['provisioningErrors']
        is_attached = obj_dict['properties']['isAttachedCompute']
        super(DataFactoryCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags,
                                                    description, created_on, modified_on, provisioning_state,
                                                    provisioning_errors, cluster_resource_id, cluster_location,
                                                    workspace, mlc_endpoint, None, workspace._auth, is_attached)

    def __repr__(self):
        """Return the string representation of the DataFactoryCompute object.

        :return: String representation of the DataFactoryCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def _create(workspace, name, provisioning_configuration):
        """Create compute target implementation method.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param provisioning_configuration:
        :type provisioning_configuration: azureml.core.compute.datafactory.DataFactoryProvisioningConfiguration
        :return:
        :rtype: azureml.core.compute.datafactory.DataFactoryCompute
        """
        create_payload = DataFactoryCompute._build_create_payload(provisioning_configuration, workspace.location)
        return ComputeTarget._create_compute_target(workspace, name, create_payload, DataFactoryCompute)

    @staticmethod
    def attach(workspace, name, resource_id):  # pragma: no cover
        """DEPRECATED. Use the ``attach_configuration`` method instead.

        Associate an existing DataFactory compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: A DataFactoryCompute object representation of the compute object.
        :rtype: azureml.core.compute.datafactory.DataFactoryCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        raise ComputeTargetException('This method is DEPRECATED. Please use the following code to attach a '
                                     'DataFactory compute resource.\n'
                                     '# Attach DataFactory\n'
                                     'attach_config = DataFactoryCompute.attach_configuration(resource_group='
                                     '"name_of_resource_group",\n'
                                     '                                                        factory_name='
                                     '"name_of_datafactory")\n'
                                     'compute = ComputeTarget.attach(workspace, name, attach_config)')

    @staticmethod
    def _attach(workspace, name, config):
        """Associates an existing DataFactory compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param config: Attach configuration object.
        :type config: azureml.core.compute.datafactory.DataFactoryAttachConfiguration
        :return: A DataFactoryCompute object representation of the compute object.
        :rtype: azureml.core.compute.datafactory.DataFactoryCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        resource_id = config.resource_id
        if not resource_id:
            resource_id = DataFactoryCompute._build_resource_id(workspace._subscription_id, config.resource_group,
                                                                config.factory_name)
        attach_payload = DataFactoryCompute._build_attach_payload(resource_id)
        return ComputeTarget._attach(workspace, name, attach_payload, DataFactoryCompute)

    @staticmethod
    def _build_resource_id(subscription_id, resource_group, factory_name):
        """Build the Azure resource ID for the compute resource.

        :param subscription_id: The Azure subscription ID
        :type subscription_id: str
        :param resource_group: Name of the resource group in which the DataFactory is located.
        :type resource_group: str
        :param factory_name: The DataFactory name
        :type factory_name: str
        :return: The Azure resource ID for the compute resource
        :rtype: str
        """
        DATAFACTORY_RESOURCE_ID_FMT = ('/subscriptions/{}/resourceGroups/{}/providers/Microsoft.DataFactory'
                                       '/factories/{}')
        return DATAFACTORY_RESOURCE_ID_FMT.format(subscription_id, resource_group, factory_name)

    @staticmethod
    def provisioning_configuration(location=None):
        """Create a configuration object for provisioning a DataFactoryCompute target.

        :param location: The location to provision cluster in. If not specified, will default to workspace location.
            Available regions for this compute can be found here:
            https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=data-factory
        :type location: str
        :return: A configuration object to be used when creating a Compute object.
        :rtype: azureml.core.compute.datafactory.DataFactoryProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        config = DataFactoryProvisioningConfiguration(location)
        return config

    @staticmethod
    def _build_create_payload(config, location):
        """Construct the payload needed to create a DataFactory.

        :param config:
        :type config: azureml.core.compute.DataFactoryProvisioningConfiguration
        :param location:
        :type location: str
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(datafactory_payload_template)
        json_payload['location'] = location
        del(json_payload['properties']['resourceId'])
        if config.location:
            json_payload['properties']['computeLocation'] = config.location
        else:
            del (json_payload['properties']['computeLocation'])
        return json_payload

    @staticmethod
    def attach_configuration(resource_group=None, factory_name=None, resource_id=None):
        """Create a configuration object for attaching a DataFactory compute target.

        :param resource_group: The name of the resource group in which the DataFactory is located.
        :type resource_group: str
        :param factory_name: The DataFactory name.
        :type factory_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: A configuration object to be used when attaching a Compute object.
        :rtype: azureml.core.compute.datafactory.DataFactoryAttachConfiguration
        """
        config = DataFactoryAttachConfiguration(resource_group, factory_name, resource_id)
        return config

    @staticmethod
    def _build_attach_payload(resource_id):
        """Build attach payload.

        :param resource_id:
        :type resource_id: str
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(datafactory_payload_template)
        json_payload['properties']['resourceId'] = resource_id
        del (json_payload['properties']['computeLocation'])
        return json_payload

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = DataFactoryCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location

    def delete(self):
        """Remove the DatafactoryCompute object from its associated workspace.

        .. remarks::

            If this object was created through Azure ML,
            the corresponding cloud based objects will also be deleted. If this object was created externally and only
            attached to the workspace, it will raise exception and nothing will be changed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('delete')

    def detach(self):
        """Detach the Datafactory object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('detach')

    def serialize(self):
        """Convert this DataFactoryCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this DataFactoryCompute object.
        :rtype: dict
        """
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': {'computeType': self.type, 'computeLocation': self.cluster_location,
                               'description': self.description,
                               'resourceId': self.cluster_resource_id,
                               'provisioningErrors': self.provisioning_errors,
                               'provisioningState': self.provisioning_state}}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into a DataFactoryCompute object.

        .. remarks::

            Will fail if the provided workspace is not the
            workspace the Compute is associated with.

        :param workspace: The workspace object the DataFactoryCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a DataFactoryCompute object.
        :type object_dict: dict
        :return: The DataFactoryCompute representation of the provided json object.
        :rtype: azureml.core.compute.datafactory.DataFactoryCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        DataFactoryCompute._validate_get_payload(object_dict)
        target = DataFactoryCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != DataFactoryCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(DataFactoryCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))


class DataFactoryProvisioningConfiguration(ComputeTargetProvisioningConfiguration):
    """Represents configuration parameters for provisioning DataFactoryCompute targets.

    Use the :meth:`azureml.core.compute.datafactory.DataFactoryCompute.provisioning_configuration` method of the
    :class:`azureml.core.compute.datafactory.DataFactoryCompute` class to
    specify provisioning parameters.

    :param location: The location to provision cluster in. If not specified, will default to workspace location.
        Available regions for this compute can be found here:
        https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=data-factory
    :type location: str
    """

    def __init__(self, location):
        """Create a configuration object for provisioning a DataFactory compute target.

        :param location: Location to provision cluster in. If not specified, will default to workspace location.
            Available regions for this compute can be found here:
            https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=data-factory
        :type location: str
        """
        super(DataFactoryProvisioningConfiguration, self).__init__(DataFactoryCompute, location)
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        pass


class DataFactoryAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching DataFactoryCompute targets.

    Use the :meth:`azureml.core.compute.datafactory.DataFactoryCompute.attach_configuration` method of the
    :class:`azureml.core.compute.datafactory.DataFactoryCompute` class to
    specify attach parameters.

    :param resource_group: The name of the resource group in which the DataFactory is located.
    :type resource_group: str
    :param factory_name: The DataFactory name.
    :type factory_name: str
    :param resource_id: The Azure resource ID for the compute resource being attached.
    :type resource_id: str
    :return: The configuration object.
    :rtype: azureml.core.compute.datafactory.DataFactoryAttachConfiguration
    """

    def __init__(self, resource_group=None, factory_name=None, resource_id=None):
        """Initialize the configuration object.

        :param resource_group: The name of the resource group in which the DataFactory is located.
        :type resource_group: str
        :param factory_name: The DataFactory name.
        :type factory_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: The configuration object.
        :rtype: azureml.core.compute.datafactory.DataFactoryAttachConfiguration
        """
        super(DataFactoryAttachConfiguration, self).__init__(DataFactoryCompute)
        self.resource_group = resource_group
        self.factory_name = factory_name
        self.resource_id = resource_id
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        if self.resource_id:
            # resource_id is provided, validate resource_id
            resource_parts = self.resource_id.split('/')
            if len(resource_parts) != 9:
                raise ComputeTargetException('Invalid resource_id provided: {}'.format(self.resource_id))
            resource_type = resource_parts[6]
            if resource_type != 'Microsoft.DataFactory':
                raise ComputeTargetException('Invalid resource_id provided, resource type {} does not match for '
                                             'DataFactory'.format(resource_type))
            # make sure do not use other info
            if self.resource_group:
                raise ComputeTargetException('Since resource_id is provided, please do not provide resource_group.')
            if self.factory_name:
                raise ComputeTargetException('Since resource_id is provided, please do not provide factory_name.')
        elif self.resource_group or self.factory_name:
            # resource_id is not provided, validate other info
            if not self.resource_group:
                raise ComputeTargetException('resource_group is not provided.')
            if not self.factory_name:
                raise ComputeTargetException('factory_name is not provided.')
        else:
            # neither resource_id nor other info is provided
            raise ComputeTargetException('Please provide resource_group and factory_name for the DataFactory compute '
                                         'resource being attached. Or please provide resource_id for the resource '
                                         'being attached.')
