# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Batch compute targets in Azure Machine Learning."""

import copy
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._util import batch_compute_template
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml.exceptions import ComputeTargetException


class BatchCompute(ComputeTarget):
    """Manages a Batch compute target in Azure Machine Learning.

    Azure Batch is used to run large-scale parallel and high-performance computing (HPC) applications
    efficiently in the cloud. BatchCompute is used in Azure  Machine Learning Pipelines to submit jobs
    to an Azure Batch pool of machines using an :class:`azureml.pipeline.steps.azurebatch_step.AzureBatchStep`.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`_

    .. remarks::

        Create an Azure Batch account before using it. To create one,
        see `Create a Batch account with the Azure
        portal <https://docs.microsoft.com/azure/batch/batch-account-create-portal>`_.

        The following example shows how to attach a Azure Batch compute account to a workspace using
        :meth:`azureml.core.compute.BatchCompute.attach_configuration`.

        .. code-block:: python

            batch_compute_name = 'mybatchcompute' # Name to associate with new compute in workspace

            # Batch account details needed to attach as compute to workspace
            batch_account_name = "<batch_account_name>" # Name of the Batch account
            batch_resource_group = "<batch_resource_group>" # Name of the resource group which contains this account

            try:
                # check if already attached
                batch_compute = BatchCompute(ws, batch_compute_name)
            except ComputeTargetException:
                print('Attaching Batch compute...')
                provisioning_config = BatchCompute.attach_configuration(resource_group=batch_resource_group,
                                                                        account_name=batch_account_name)
                batch_compute = ComputeTarget.attach(ws, batch_compute_name, provisioning_config)
                batch_compute.wait_for_completion()
                print("Provisioning state:{}".format(batch_compute.provisioning_state))
                print("Provisioning errors:{}".format(batch_compute.provisioning_errors))

            print("Using Batch compute:{}".format(batch_compute.cluster_resource_id))

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-how-to-use-azurebatch-to-run-a-windows-executable.ipynb


    :param workspace: The workspace object containing the BatchCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the BatchCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'Batch'

    def _initialize(self, workspace, obj_dict):
        """
        Initialize abstract method.

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
        created_on = None
        modified_on = None
        cluster_resource_id = obj_dict['properties']['resourceId']
        cluster_location = obj_dict['properties']['computeLocation'] \
            if 'computeLocation' in obj_dict['properties'] else location
        provisioning_state = obj_dict['properties']['provisioningState']
        provisioning_errors = obj_dict['properties']['provisioningErrors']
        is_attached = obj_dict['properties']['isAttachedCompute']
        super(BatchCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags,
                                              description, created_on, modified_on, provisioning_state,
                                              provisioning_errors, cluster_resource_id, cluster_location,
                                              workspace, mlc_endpoint, None, workspace._auth, is_attached)

    def __repr__(self):
        """Return the string representation of the BatchCompute object.

        :return: String representation of the BatchCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def _attach(workspace, name, config):
        """
        Associates an existing Batch compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param config: Attach configuration object.
        :type config: azureml.core.compute.batch.BatchAttachConfiguration
        :return: A BatchCompute object representation of the compute object.
        :rtype: azureml.core.compute.batch.BatchCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        resource_id = config.resource_id
        if not resource_id:
            resource_id = BatchCompute._build_resource_id(workspace._subscription_id, config.resource_group,
                                                          config.account_name)
        attach_payload = BatchCompute._build_attach_payload(resource_id)
        return ComputeTarget._attach(workspace, name, attach_payload, BatchCompute)

    @staticmethod
    def _build_resource_id(subscription_id, resource_group, account_name):
        """
        Build the Azure resource ID for the compute resource.

        :param subscription_id: The Azure subscription ID
        :type subscription_id: str
        :param resource_group: Name of the resource group in which the Batch account is located.
        :type resource_group: str
        :param account_name: The Batch account name
        :type account_name: str
        :return: The Azure resource ID for the compute resource
        :rtype: str
        """
        BATCH_RESOURCE_ID_FMT = '/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Batch/batchAccounts/{}'
        return BATCH_RESOURCE_ID_FMT.format(subscription_id, resource_group, account_name)

    @staticmethod
    def attach_configuration(resource_group=None, account_name=None, resource_id=None):
        """Create a configuration object for attaching a Batch compute target.

        :param resource_group: The name of the resource group in which the Batch account is located.
        :type resource_group: str
        :param account_name: The Batch account name.
        :type account_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: A configuration object to be used when attaching a Compute object.
        :rtype: azureml.core.compute.batch.BatchAttachConfiguration
        """
        config = BatchAttachConfiguration(resource_group, account_name, resource_id)
        return config

    @staticmethod
    def _build_attach_payload(resource_id):
        """Build attach payload.

        :param resource_id:
        :type resource_id: str
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(batch_compute_template)
        json_payload['properties']['resourceId'] = resource_id
        return json_payload

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = BatchCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location

    def delete(self):
        """
        Delete is not supported for a BatchCompute object. Use :meth:`detach` instead.

        :raises azureml.exceptions.ComputeTargetException:
        """
        raise ComputeTargetException('Delete is not supported for Batch object. Try to use detach instead.')

    def detach(self):
        """
        Detaches the Batch object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('detach')

    def serialize(self):
        """
        Convert this BatchCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this BatchCompute object.
        :rtype: dict
        """
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': {'computeType': 'Batch', 'computeLocation': self.cluster_location,
                               'description': self.description,
                               'resourceId': self.cluster_resource_id,
                               'provisioningErrors': self.provisioning_errors,
                               'provisioningState': self.provisioning_state}}

    @staticmethod
    def deserialize(workspace, object_dict):
        """
        Convert a JSON object into a BatchCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided workspace is not the
            workspace the Compute is associated with.

        :param workspace: The workspace object the BatchCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a BatchCompute object.
        :type object_dict: dict
        :return: The BatchCompute representation of the provided JSON object.
        :rtype: azureml.core.compute.batch.BatchCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        BatchCompute._validate_get_payload(object_dict)
        target = BatchCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != BatchCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(BatchCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))


class BatchAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching BatchCompute targets.

    Use the :meth:`azureml.core.compute.batch.BatchCompute.attach_configuration` method of the
    :class:`azureml.core.compute.batch.BatchCompute` class to
    specify attach parameters.

    :param resource_group: The name of the resource group in which the Batch account is located.
    :type resource_group: str
    :param account_name: The Batch account name.
    :type account_name: str
    :param resource_id: The Azure resource ID for the compute resource being attached.
    :type resource_id: str
    :return: The configuration object.
    :rtype: azureml.core.compute.batch.BatchAttachConfiguration
    """

    def __init__(self, resource_group=None, account_name=None, resource_id=None):
        """Initialize the configuration object.

        :param resource_group: Name of the resource group in which the Batch account is located.
        :type resource_group: str
        :param account_name: The Batch account name
        :type account_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached
        :type resource_id: str
        :return: The configuration object
        :rtype: azureml.core.compute.batch.BatchAttachConfiguration
        """
        super(BatchAttachConfiguration, self).__init__(BatchCompute)
        self.resource_group = resource_group
        self.account_name = account_name
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
            if resource_type != 'Microsoft.Batch':
                raise ComputeTargetException('Invalid resource_id provided, resource type {} does not match for '
                                             'Batch'.format(resource_type))
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
            raise ComputeTargetException('Please provide resource_group and account_name for the Batch compute '
                                         'resource being attached. Or please provide resource_id for the resource '
                                         'being attached.')
