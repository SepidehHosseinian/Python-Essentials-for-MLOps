# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Azure Data Lake Analytics compute targets in Azure Machine Learning."""

import copy
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._util import adla_payload_template
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml.exceptions import ComputeTargetException


class AdlaCompute(ComputeTarget):
    """Manages an Azure Data Lake Analytics compute target in Azure Machine Learning.

    Azure Data Lake Analytics is a big data analytics platform in the Azure cloud. It can be used as a compute target
    with an Azure Machine Learning pipelines.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`_

    .. remarks::

        Create an Azure Data Lake Analytics account before using it. To create one,
        see `Get started with Azure Data Lake
        Analytics <https://docs.microsoft.com/azure/data-lake-analytics/data-lake-analytics-get-started-portal>`_.

        The following example shows how to attach an ADLA account to a workspace using the
        :meth:`azureml.core.compute.AdlaCompute.attach_configuration` method.

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


    :param workspace: The workspace object containing the AdlaCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the AdlaCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'DataLakeAnalytics'

    def _initialize(self, workspace, obj_dict):
        """Class AdlaCompute constructor.

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
        super(AdlaCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags,
                                             description, created_on, modified_on, provisioning_state,
                                             provisioning_errors, cluster_resource_id, cluster_location,
                                             workspace, mlc_endpoint, None, workspace._auth, is_attached)

    def __repr__(self):
        """Return the string representation of the AdlaCompute object.

        :return: String representation of the AdlaCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def attach(workspace, name, resource_id):  # pragma: no cover
        """DEPRECATED. Use the ``attach_configuration`` method instead.

        Associate an existing Azure Data Lake Analytics compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: An AdlaCompute object representation of the compute object.
        :rtype: azureml.core.compute.adla.AdlaCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        raise ComputeTargetException('This method is DEPRECATED. Please use the following code to attach a ADLA '
                                     'compute resource.\n'
                                     '# Attach ADLA\n'
                                     'attach_config = AdlaCompute.attach_configuration(resource_group='
                                     '"name_of_resource_group",\n'
                                     '                                                 account_name='
                                     '"name_of_adla_account")\n'
                                     'compute = ComputeTarget.attach(workspace, name, attach_config)')

    @staticmethod
    def _attach(workspace, name, config):
        """Associates an existing Azure Data Lake Analytics compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param config: Attach configuration object
        :type config: azureml.core.compute.adla.DataLakeAnalyticsAttachConfiguration
        :return: An AdlaCompute object representation of the compute object.
        :rtype: azureml.core.compute.adla.AdlaCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        resource_id = config.resource_id
        if not resource_id:
            resource_id = AdlaCompute._build_resource_id(workspace._subscription_id, config.resource_group,
                                                         config.account_name)
        attach_payload = AdlaCompute._build_attach_payload(resource_id)
        return ComputeTarget._attach(workspace, name, attach_payload, AdlaCompute)

    @staticmethod
    def _build_resource_id(subscription_id, resource_group, account_name):
        """Build the Azure resource ID for the compute resource.

        :param subscription_id: The Azure subscription ID
        :type subscription_id: str
        :param resource_group: Name of the resource group in which the DataLakeAnalytics is located.
        :type resource_group: str
        :param account_name: The DataLakeAnalytics account name
        :type account_name: str
        :return: The Azure resource ID for the compute resource
        :rtype: str
        """
        ADLA_RESOURCE_ID_FMT = ('/subscriptions/{}/resourceGroups/{}/providers/Microsoft.DataLakeAnalytics/'
                                'accounts/{}')
        return ADLA_RESOURCE_ID_FMT.format(subscription_id, resource_group, account_name)

    @staticmethod
    def attach_configuration(resource_group=None, account_name=None, resource_id=None):
        """Create a configuration object for attaching an Azure Data Lake Analytics compute target.

        :param resource_group: The name of the resource group in which the Data Lake Analytics account is located.
        :type resource_group: str
        :param account_name: The Data Lake Analytics account name.
        :type account_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: A configuration object to be used when attaching a compute object.
        :rtype: azureml.core.compute.adla.DataLakeAnalyticsAttachConfiguration
        """
        config = DataLakeAnalyticsAttachConfiguration(resource_group, account_name, resource_id)
        return config

    @staticmethod
    def _build_attach_payload(resource_id):
        """Build attach payload.

        :param resource_id:
        :type resource_id: str
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(adla_payload_template)
        json_payload['properties']['resourceId'] = resource_id
        del (json_payload['properties']['computeLocation'])
        return json_payload

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = AdlaCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location

    def delete(self):
        """Remove the AdlaCompute object from its associated workspace.

        If this object was created through Azure Machine Learning, the corresponding cloud based objects will
        also be deleted. If this object was created externally and only attached to the workspace, it raises
        a :class:`azureml.exceptions.ComputeTargetException` and nothing is changed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('delete')

    def detach(self):
        """Detach the AdlaCompute object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('detach')

    def serialize(self):
        """Convert this AdlaCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this AdlaCompute object.
        :rtype: dict
        """
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': {'computeType': self.type, 'computeLocation': self.cluster_location,
                               'description': self.description, 'resourceId': self.cluster_resource_id,
                               'isAttachedCompute': self.is_attached,
                               'provisioningErrors': self.provisioning_errors,
                               'provisioningState': self.provisioning_state}}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into an AdlaCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the AdlaCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to an AdlaCompute object.
        :type object_dict: dict
        :return: The AdlaCompute representation of the provided JSON object.
        :rtype: azureml.core.compute.adla.AdlaCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        AdlaCompute._validate_get_payload(object_dict)
        target = AdlaCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != AdlaCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(AdlaCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))


class DataLakeAnalyticsAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching AdlaCompute targets.

    Use the :meth:`azureml.core.compute.adla.AdlaCompute.attach_configuration` method of the
    :class:`azureml.core.compute.adla.AdlaCompute` class to
    specify attach parameters.
    """

    def __init__(self, resource_group=None, account_name=None, resource_id=None):
        """Initialize the configuration object.

        :param resource_group: Name of the resource group in which the Data Lake Analytics is located.
        :type resource_group: str
        :param account_name: The Data Lake Analytics account name.
        :type account_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: The configuration object.
        :rtype: azureml.core.compute.adla.DataLakeAnalyticsAttachConfiguration
        """
        super(DataLakeAnalyticsAttachConfiguration, self).__init__(AdlaCompute)
        self.resource_group = resource_group
        self.account_name = account_name
        self.resource_id = resource_id
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        :raises azureml.exceptions.ComputeTargetException: Configuration validation failed.
        """
        if self.resource_id:
            # resource_id is provided, validate resource_id
            resource_parts = self.resource_id.split('/')
            if len(resource_parts) != 9:
                raise ComputeTargetException('Invalid resource_id provided: {}'.format(self.resource_id))
            resource_type = resource_parts[6]
            if resource_type != 'Microsoft.DataLakeAnalytics':
                raise ComputeTargetException('Invalid resource_id provided, resource type {} does not match for '
                                             'DataLakeAnalytics'.format(resource_type))
            # make sure do not use other info
            if self.resource_group:
                raise ComputeTargetException('Since resource_id is provided, please do not provide resource_group.')
            if self.account_name:
                raise ComputeTargetException('Since resource_id is provided, please do not provide account_name.')
        elif self.resource_group or self.account_name:
            # resource_id is not provided, validate other info
            if not self.resource_group:
                raise ComputeTargetException('resource_group is not provided.')
            if not self.account_name:
                raise ComputeTargetException('account_name is not provided.')
        else:
            # neither resource_id nor other info is provided
            raise ComputeTargetException('Please provide resource_group and account_name for the ADLA compute '
                                         'resource being attached. Or please provide resource_id for the resource '
                                         'being attached.')
