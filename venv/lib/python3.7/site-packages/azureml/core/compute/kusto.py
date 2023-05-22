# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Kusto compute targets in Azure Machine Learning."""

import copy
import json
import requests
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml._compute._util import kusto_compute_template
from azureml._compute._util import get_requests_session
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml.exceptions import ComputeTargetException
from azureml._restclient.clientbase import ClientBase


class KustoCompute(ComputeTarget):
    """
    Manages a Kusto compute target in Azure Machine Learning.

    Kusto, also known as Azure Data Explorer, can be used as a compute target with an
    Azure Machine Learning pipeline.
    The compute target holds the Kusto connection string and service principal credentials
    used to access the target Kusto cluster.

    :param workspace: The workspace object containing the KustoCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the KustoCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'Kusto'

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
        super(KustoCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags,
                                              description, created_on, modified_on, provisioning_state,
                                              provisioning_errors, cluster_resource_id, cluster_location,
                                              workspace, mlc_endpoint, None, workspace._auth, is_attached)

    def __repr__(self):
        """Return the string representation of the KustoCompute object.

        :return: String representation of the KustoCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def _attach(workspace, name, config):
        """Associates an existing Kusto compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param config: Attach configuration object.
        :type config: azureml.core.compute.kusto.KustoAttachConfiguration
        :return: A KustoCompute object representation of the compute object
        :rtype: azureml.core.compute.kusto.KustoCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        attach_payload = KustoCompute._build_attach_payload(location=workspace.location,
                                                            resource_id=config.resource_id,
                                                            kusto_connection_string=config.kusto_connection_string,
                                                            client_id=config.application_id,
                                                            secret=config.application_key,
                                                            tenant_id=config.tenant_id)
        return ComputeTarget._attach(workspace, name, attach_payload, KustoCompute)

    @staticmethod
    def attach_configuration(resource_group=None, workspace_name=None, resource_id=None, tenant_id=None,
                             kusto_connection_string=None, application_id=None, application_key=None):
        """Create a configuration object for attaching a Kusto compute target.

        :param resource_group: The name of the resource group of the workspace.
        :type resource_group: str
        :param workspace_name: The workspace name.
        :type workspace_name: str
        :param resource_id: The Azure resource ID of the compute resource.
        :type resource_id: str
        :param tenant_id: The tenant ID of the compute resource.
        :type tenant_id: str
        :param kusto_connection_string: The connection string of the Kusto cluster.
        :type kusto_connection_string: str
        :param application_id: The application ID of the compute resource.
        :type application_id: str
        :param application_key: The application key of the compute resource.
        :type application_key: str
        :return: A configuration object to be used when attaching a Compute object.
        :rtype: azureml.core.compute.kusto.KustoAttachConfiguration
        """
        config = KustoAttachConfiguration(resource_group=resource_group, workspace_name=workspace_name,
                                          resource_id=resource_id, tenant_id=tenant_id,
                                          kusto_connection_string=kusto_connection_string,
                                          application_id=application_id,
                                          application_key=application_key)
        return config

    @staticmethod
    def _build_attach_payload(location, resource_id, kusto_connection_string, client_id, secret, tenant_id):
        """Build attach payload.

        :param location: Location of the workspace
        :type location: str
        :param resource_id: The Azure resource ID of the compute resource.
        :type resource_id: str
        :param kusto_connection_string: The connection string of the Kusto cluster.
        :type kusto_connection_string: str
        :param client_id: The application ID of the compute resource.
        :type client_id: str
        :param secret: The application key of the compute resource.
        :type secret: str
        :param tenant_id: The tenant ID of the compute resource.
        :type tenant_id: str
        :return: The payload for the attach request
        :rtype: dict[str, Any]
        """
        json_payload = copy.deepcopy(kusto_compute_template)
        json_payload['location'] = location
        json_payload['properties']['resourceId'] = resource_id
        json_payload['properties']['properties']['kustoConnectionString'] = kusto_connection_string
        json_payload['properties']['properties']['servicePrincipalCredentials']['clientId'] = client_id
        json_payload['properties']['properties']['servicePrincipalCredentials']['secret'] = secret
        json_payload['properties']['properties']['servicePrincipalCredentials']['tenantId'] = tenant_id
        return json_payload

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = KustoCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location

    def delete(self):
        """Delete is not supported for a KustoCompute object. Use :meth:`detach` instead.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        raise ComputeTargetException('Delete is not supported for Kusto object. Try to use detach instead.')

    def detach(self):
        """Detaches the Kusto object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        self._delete_or_detach('detach')

    def get_credentials(self):
        """Retrieve the credentials for the Kusto target.

        :return: The credentials for the Kusto target.
        :rtype: dict[str, dict[str, str]]
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
        """Convert this KustoCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this KustoCompute object.
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
        """Convert a JSON object into a KustoCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the KustoCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a KustoCompute object.
        :type object_dict: dict
        :return: The KustoCompute representation of the provided JSON object.
        :rtype: azureml.core.compute.kusto.KustoCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        KustoCompute._validate_get_payload(object_dict)
        target = KustoCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != KustoCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(KustoCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))


class KustoAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching KustoCompute targets.

    Use the :meth:`azureml.core.compute.kusto.KustoCompute.attach_configuration` method of the
    :class:`azureml.core.compute.kusto.KustoCompute` class to
    specify attach parameters.

    """

    def __init__(self, resource_group=None, workspace_name=None, resource_id=None, tenant_id=None,
                 kusto_connection_string=None, application_id=None, application_key=None):
        """Initialize the configuration object.

        :param resource_group: The name of the resource group of the workspace.
        :type resource_group: str
        :param workspace_name: The workspace name.
        :type workspace_name: str
        :param resource_id: The Azure resource ID of the compute resource.
        :type resource_id: str
        :param tenant_id: The tenant ID of the compute resource.
        :type tenant_id: str
        :param kusto_connection_string: The connection string of the Kusto cluster.
        :type kusto_connection_string: str
        :param application_id: The application ID of the compute resource.
        :type application_id: str
        :param application_key: The application key of the compute resource.
        :type application_key: str
        :return: The configuration object.
        :rtype: azureml.core.compute.kusto.KustoAttachConfiguration
        """
        super(KustoAttachConfiguration, self).__init__(KustoCompute)
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.resource_id = resource_id
        self.tenant_id = tenant_id
        self.kusto_connection_string = kusto_connection_string
        self.application_id = application_id
        self.application_key = application_key

        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        if self.resource_id:
            resource_parts = self.resource_id.split('/')
            if len(resource_parts) != 9:
                raise ComputeTargetException('Invalid resource_id provided: {}'.format(self.resource_id))
            resource_type = resource_parts[6]
            if resource_type != 'Microsoft.Kusto':
                raise ComputeTargetException('Invalid resource_id provided, resource type {} does not match for '
                                             'Kusto'.format(resource_type))
        if not self.resource_group or not self.workspace_name or not self.resource_id:
            raise ComputeTargetException('Please provide resource group, workspace name, and resource id')
        if not self.tenant_id or not self.application_key or not self.application_id:
            raise ComputeTargetException('Please provide tenant id, application id, and application key')
        if not self.kusto_connection_string:
            raise ComputeTargetException('Please provide kusto connection string for the target cluster')
