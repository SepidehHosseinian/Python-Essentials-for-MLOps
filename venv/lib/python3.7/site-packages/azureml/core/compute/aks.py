# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Azure Kubernetes Service compute targets in Azure Machine Learning."""

import copy
import json
import requests
import traceback
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml._compute._util import aks_payload_template
from azureml._compute._util import get_requests_session
from azureml.core.compute import ComputeTarget
from azureml.core.compute.compute import ComputeTargetProvisioningConfiguration
from azureml.core.compute.compute import ComputeTargetAttachConfiguration
from azureml.core.compute.compute import ComputeTargetUpdateConfiguration
from azureml.exceptions import ComputeTargetException
from azureml._restclient.clientbase import ClientBase


class AksCompute(ComputeTarget):
    """Manages an Azure Kubernetes Service compute target in Azure Machine Learning.

    Azure Kubernetes Service (AKSCompute) targets are typically used for high-scale production deployments because
    they provides fast response time and autoscaling of the deployed service.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`_

    .. remarks::

        The following sample shows how to create an AKS cluster with FPGA-enabled machines.

        .. code-block:: python

            from azureml.core.compute import AksCompute, ComputeTarget

            # Uses the specific FPGA enabled VM (sku: Standard_PB6s)
            # Standard_PB6s are available in: eastus, westus2, westeurope, southeastasia
            prov_config = AksCompute.provisioning_configuration(vm_size = "Standard_PB6s",
                                                                agent_count = 1,
                                                                location = "eastus")

            aks_name = 'my-aks-pb6'
            # Create the cluster
            aks_target = ComputeTarget.create(workspace = ws,
                                              name = aks_name,
                                              provisioning_configuration = prov_config)

    :param workspace: The workspace object containing the AksCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the AksCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'AKS'

    def _initialize(self, workspace, obj_dict):
        """Class AksCompute constructor.

        :param workspace: The workspace object containing the Compute object to retrieve.
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
        aks_properties = obj_dict['properties']['properties']
        agent_vm_size = aks_properties['agentVmSize'] if aks_properties else None
        agent_count = aks_properties['agentCount'] if aks_properties else None
        cluster_purpose = aks_properties['clusterPurpose'] if aks_properties else None
        cluster_fqdn = aks_properties['clusterFqdn'] if aks_properties else None
        load_balancer_type = aks_properties['loadBalancerType'] if aks_properties else None
        load_balancer_subnet = aks_properties['loadBalancerSubnet'] if aks_properties else None
        system_services = aks_properties['systemServices'] if aks_properties else None
        if system_services:
            system_services = [SystemService.deserialize(service) for service in system_services]
        ssl_configuration = aks_properties['sslConfiguration'] \
            if aks_properties and 'sslConfiguration' in aks_properties else None
        if ssl_configuration:
            ssl_configuration = SslConfiguration.deserialize(ssl_configuration)
        super(AksCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags, description,
                                            created_on, modified_on, provisioning_state, provisioning_errors,
                                            cluster_resource_id, cluster_location, workspace, mlc_endpoint, None,
                                            workspace._auth, is_attached)
        self.agent_vm_size = agent_vm_size
        self.agent_count = agent_count
        self.cluster_purpose = cluster_purpose
        self.cluster_fqdn = cluster_fqdn
        self.system_services = system_services
        self.ssl_configuration = ssl_configuration
        self.load_balancer_type = load_balancer_type
        self.load_balancer_subnet = load_balancer_subnet

    def __repr__(self):
        """Return the string representation of the AksCompute object.

        :return: String representation of the AksCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def _create(workspace, name, provisioning_configuration):
        """Create compute.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param provisioning_configuration:
        :type provisioning_configuration: AksProvisioningConfiguration
        :return:
        :rtype: azureml.core.compute.aks.AksCompute
        """
        compute_create_payload = AksCompute._build_create_payload(provisioning_configuration,
                                                                  workspace.location, workspace.subscription_id)
        return ComputeTarget._create_compute_target(workspace, name, compute_create_payload, AksCompute)

    @staticmethod
    def attach(workspace, name, resource_id):  # pragma: no cover
        """DEPRECATED. Use the ``attach_configuration`` method instead.

        Associate an existing AKS compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :return: An AksCompute object representation of the compute object.
        :rtype: azureml.core.compute.aks.AksCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        raise ComputeTargetException('This method is DEPRECATED. Please use the following code to attach a AKS '
                                     'compute resource.\n'
                                     '# Attach AKS\n'
                                     'attach_config = AksCompute.attach_configuration(resource_group='
                                     '"name_of_resource_group",\n'
                                     '                                                cluster_name='
                                     '"name_of_aks_cluster")\n'
                                     'compute = ComputeTarget.attach(workspace, name, attach_config)')

    @staticmethod
    def _attach(workspace, name, config):
        """Associate an existing AKS compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached.
        :type name: str
        :param config: Attach configuration object
        :type config: azureml.core.compute.aks.AksAttachConfiguration
        :return: An AksCompute object representation of the compute object
        :rtype: azureml.core.compute.aks.AksCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        resource_id = config.resource_id
        if not resource_id:
            resource_id = AksCompute._build_resource_id(workspace._subscription_id, config.resource_group,
                                                        config.cluster_name)
        attach_payload = AksCompute._build_attach_payload(config, resource_id)
        return ComputeTarget._attach(workspace, name, attach_payload, AksCompute)

    @staticmethod
    def _build_resource_id(subscription_id, resource_group, cluster_name):
        """Build the Azure resource ID for the compute resource.

        :param subscription_id: The Azure subscription ID
        :type subscription_id: str
        :param resource_group: Name of the resource group in which the AKS is located.
        :type resource_group: str
        :param cluster_name: The AKS cluster name
        :type cluster_name: str
        :return: The Azure resource ID for the compute resource
        :rtype: str
        """
        AKS_RESOURCE_ID_FMT = ('/subscriptions/{}/resourcegroups/{}/providers/Microsoft.ContainerService/'
                               'managedClusters/{}')
        return AKS_RESOURCE_ID_FMT.format(subscription_id, resource_group, cluster_name)

    @staticmethod
    def provisioning_configuration(agent_count=None, vm_size=None, ssl_cname=None, ssl_cert_pem_file=None,
                                   ssl_key_pem_file=None, location=None, vnet_resourcegroup_name=None,
                                   vnet_name=None, subnet_name=None, service_cidr=None,
                                   dns_service_ip=None, docker_bridge_cidr=None, cluster_purpose=None,
                                   load_balancer_type=None, load_balancer_subnet=None):
        """Create a configuration object for provisioning an AKS compute target.

        :param agent_count: The number of agents (VMs) to host containers. Defaults to 3.
        :type agent_count: int
        :param vm_size: The size of agent VMs. A full list of options can be found here:
            https://aka.ms/azureml-aks-details. Defaults to Standard_D3_v2.
        :type vm_size: str
        :param ssl_cname: A CName to use if enabling SSL validation on the cluster. Must provide all three
            CName, cert file, and key file to enable SSL validation.
        :type ssl_cname: str
        :param ssl_cert_pem_file: A file path to a file containing cert information for SSL validation. Must provide
            all three CName, cert file, and key file to enable SSL validation.
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: A file path to a file containing key information for SSL validation. Must provide
            all three CName, cert file, and key file to enable SSL validation.
        :type ssl_key_pem_file: str
        :param location: The location to provision cluster in. If not specified, will default to workspace location.
            Available regions for this compute can be found here:
            https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=kubernetes-service
        :type location: str
        :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located.
        :type vnet_resourcegroup_name: str
        :param vnet_name: The name of the virtual network.
        :type vnet_name: str
        :param subnet_name: The name of the subnet inside the vnet.
        :type subnet_name: str
        :param service_cidr: A CIDR notation IP range from which to assign service cluster IPs.
        :type service_cidr: str
        :param dns_service_ip: Containers DNS server IP address.
        :type dns_service_ip: str
        :param docker_bridge_cidr: A CIDR notation IP for Docker bridge.
        :type docker_bridge_cidr: str
        :param cluster_purpose: Targeted usage of the cluster. This is used to provision Azure Machine Learning
            components to ensure the desired level of fault-tolerance and QoS. AksCompute.ClusterPurpose class
            is provided for convenience of specifying available values. More detailed information of these values
            and their use cases can be found here: https://aka.ms/azureml-create-attach-aks
        :type cluster_purpose: str
        :param load_balancer_type: Load balancer type of AKS cluster.
            Valid values are PublicIp and InternalLoadBalancer. Default value is PublicIp.
        :type load_balancer_type: str
        :param load_balancer_subnet: Load balancer subnet of AKS cluster.
            It can be used only when Internal Load Balancer is used as load balancer type. Default value is aks-subnet.
        :type load_balancer_subnet: str
        :return: A configuration object to be used when creating a Compute object
        :rtype: azureml.core.compute.aks.AksProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        config = AksProvisioningConfiguration(agent_count, vm_size, ssl_cname, ssl_cert_pem_file, ssl_key_pem_file,
                                              location, vnet_resourcegroup_name, vnet_name, subnet_name, service_cidr,
                                              dns_service_ip, docker_bridge_cidr, cluster_purpose, load_balancer_type,
                                              load_balancer_subnet)
        return config

    @staticmethod
    def _build_create_payload(config, location, subscription_id):
        """Construct the payload needed to create an AKS cluster.

        :param config:
        :type config: azureml.core.compute.aks.AksProvisioningConfiguration
        :param location:
        :type location:
        :param subscription_id:
        :type subscription_id:
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(aks_payload_template)
        del(json_payload['properties']['resourceId'])
        json_payload['location'] = location
        if not config.agent_count and not config.vm_size and not config.ssl_cname and not config.vnet_name and \
                not config.vnet_resourcegroup_name and not config.subnet_name and not config.service_cidr and \
                not config.dns_service_ip and not config.docker_bridge_cidr and not config.leaf_domain_label and \
                not config.cluster_purpose and not config.load_balancer_type and not config.load_balancer_subnet:
            del(json_payload['properties']['properties'])
        else:
            if config.agent_count:
                json_payload['properties']['properties']['agentCount'] = config.agent_count
            else:
                del(json_payload['properties']['properties']['agentCount'])
            if config.vm_size:
                json_payload['properties']['properties']['agentVmSize'] = config.vm_size
            else:
                del(json_payload['properties']['properties']['agentVmSize'])
            if config.cluster_purpose:
                json_payload['properties']['properties']['clusterPurpose'] = config.cluster_purpose
            else:
                del(json_payload['properties']['properties']['clusterPurpose'])
            if config.load_balancer_type:
                json_payload['properties']['properties']['loadBalancerType'] = config.load_balancer_type
            else:
                del(json_payload['properties']['properties']['loadBalancerType'])
            if config.load_balancer_subnet:
                json_payload['properties']['properties']['loadBalancerSubnet'] = config.load_balancer_subnet
            else:
                del(json_payload['properties']['properties']['loadBalancerSubnet'])
            if config.ssl_cname:
                try:
                    with open(config.ssl_cert_pem_file, 'r') as cert_file:
                        cert_data = cert_file.read()
                    with open(config.ssl_key_pem_file, 'r') as key_file:
                        key_data = key_file.read()
                except (IOError, OSError):
                    raise ComputeTargetException("Error while reading ssl information:\n"
                                                 "{}".format(traceback.format_exc().splitlines()[-1]))
                json_payload['properties']['properties']['sslConfiguration']['status'] = 'Enabled'
                json_payload['properties']['properties']['sslConfiguration']['cname'] = config.ssl_cname
                json_payload['properties']['properties']['sslConfiguration']['cert'] = cert_data
                json_payload['properties']['properties']['sslConfiguration']['key'] = key_data
                del(json_payload['properties']['properties']['sslConfiguration']['leafDomainLabel'])
                del(json_payload['properties']['properties']['sslConfiguration']['overwriteExistingDomain'])
            elif config.leaf_domain_label:
                json_payload['properties']['properties']['sslConfiguration']['status'] = 'Auto'
                json_payload['properties']['properties']['sslConfiguration']['leafDomainLabel'] = \
                    config.leaf_domain_label
                json_payload['properties']['properties']['sslConfiguration']['overwriteExistingDomain'] = \
                    config.overwrite_existing_domain
                del(json_payload['properties']['properties']['sslConfiguration']['cname'])
                del(json_payload['properties']['properties']['sslConfiguration']['cert'])
                del(json_payload['properties']['properties']['sslConfiguration']['key'])
            else:
                del(json_payload['properties']['properties']['sslConfiguration'])
            if config.vnet_name:
                json_payload['properties']['properties']['aksNetworkingConfiguration']['subnetId'] = \
                    "/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks" \
                    "/{2}/subnets/{3}".format(subscription_id, config.vnet_resourcegroup_name,
                                              config.vnet_name, config.subnet_name)
                json_payload['properties']['properties']['aksNetworkingConfiguration']['serviceCidr'] = \
                    config.service_cidr
                json_payload['properties']['properties']['aksNetworkingConfiguration']['dnsServiceIP'] = \
                    config.dns_service_ip
                json_payload['properties']['properties']['aksNetworkingConfiguration']['dockerBridgeCidr'] = \
                    config.docker_bridge_cidr
            else:
                del(json_payload['properties']['properties']['aksNetworkingConfiguration'])
        if config.location:
            json_payload['properties']['computeLocation'] = config.location
        else:
            del(json_payload['properties']['computeLocation'])
        return json_payload

    @staticmethod
    def attach_configuration(
            resource_group=None, cluster_name=None, resource_id=None, cluster_purpose=None,
            load_balancer_type=None, load_balancer_subnet=None
    ):
        """Create a configuration object for attaching a AKS compute target.

        :param resource_group: The name of the resource group in which the AKS is located.
        :type resource_group: str
        :param cluster_name: The AKS cluster name.
        :type cluster_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :param cluster_purpose: The targeted usage of the cluster. This is used to provision Azure Machine
          Learning components to ensure the desired level of fault-tolerance and QoS. The
          :class:`azureml.core.compute.aks.AksCompute.ClusterPurpose` class defines the possible values.
          For more information, see `Attach an existing AKS
          cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.
        :type cluster_purpose: str
        :param load_balancer_type: The AKS cluster type. Valid values are PublicIp and InternalLoadBalancer.
          Default value is PublicIp.
        :type load_balancer_type: str
        :param load_balancer_subnet: The AKS load balancer subnet. It can be used only when InternalLoadBalancer
          is used as load balancer type. Default value is aks-subnet.
        :type load_balancer_subnet: str
        :return: A configuration object to be used when attaching a Compute object.
        :rtype: azureml.core.compute.aks.AksAttachConfiguration
        """
        config = AksAttachConfiguration(
            resource_group, cluster_name, resource_id, cluster_purpose, load_balancer_type,
            load_balancer_subnet
        )
        return config

    @staticmethod
    def _build_attach_payload(config, resource_id):
        """Build attach payload.

        :param config: Attach configuration object
        :type config: azureml.core.compute.aks.AksAttachConfiguration
        :param resource_id:
        :type resource_id: str
        :return:
        :rtype: dict
        """
        json_payload = copy.deepcopy(aks_payload_template)
        json_payload['properties']['resourceId'] = resource_id
        del(json_payload['properties']['computeLocation'])
        del(json_payload['properties']['properties']['agentCount'])
        del(json_payload['properties']['properties']['agentVmSize'])
        if config.cluster_purpose:
            json_payload['properties']['properties']['clusterPurpose'] = config.cluster_purpose
        else:
            del(json_payload['properties']['properties']['clusterPurpose'])
        del(json_payload['properties']['properties']['aksNetworkingConfiguration'])
        if config.ssl_cname:
            try:
                with open(config.ssl_cert_pem_file, 'r') as cert_file:
                    cert_data = cert_file.read()
                with open(config.ssl_key_pem_file, 'r') as key_file:
                    key_data = key_file.read()
            except (IOError, OSError):
                raise ComputeTargetException("Error while reading ssl information:\n"
                                             "{}".format(traceback.format_exc().splitlines()[-1]))
            json_payload['properties']['properties']['sslConfiguration']['status'] = 'Enabled'
            json_payload['properties']['properties']['sslConfiguration']['cname'] = config.ssl_cname
            json_payload['properties']['properties']['sslConfiguration']['cert'] = cert_data
            json_payload['properties']['properties']['sslConfiguration']['key'] = key_data
            del(json_payload['properties']['properties']['sslConfiguration']['leafDomainLabel'])
            del(json_payload['properties']['properties']['sslConfiguration']['overwriteExistingDomain'])
        elif config.leaf_domain_label:
            json_payload['properties']['properties']['sslConfiguration']['status'] = 'Auto'
            json_payload['properties']['properties']['sslConfiguration']['leafDomainLabel'] = \
                config.leaf_domain_label
            json_payload['properties']['properties']['sslConfiguration']['overwriteExistingDomain'] = \
                config.overwrite_existing_domain
            del(json_payload['properties']['properties']['sslConfiguration']['cname'])
            del(json_payload['properties']['properties']['sslConfiguration']['cert'])
            del(json_payload['properties']['properties']['sslConfiguration']['key'])
        else:
            del(json_payload['properties']['properties']['sslConfiguration'])

        if config.load_balancer_type:
            json_payload['properties']['properties']['loadBalancerType'] = config.load_balancer_type
        if config.load_balancer_subnet:
            json_payload['properties']['properties']['loadBalancerSubnet'] = config.load_balancer_subnet

        if not json_payload['properties']['properties']:
            del(json_payload['properties']['properties'])

        return json_payload

    def _build_update_payload(self, config):
        """Build update payload.

        :param config: Update configuration object
        :type config: azureml.core.compute.aks.AksUpdateConfiguration
        :return:
        :rtype: dict
        """
        json_payload = self._get(self.workspace, self.name)
        if json_payload is None:
            raise ComputeTargetException('ComputeTargetNotFound: Compute Target with name {} not found in '
                                         'provided workspace'.format(self.name))

        if config.ssl_configuration and config.ssl_configuration.status == 'Disabled':
            json_payload['properties']['properties']['sslConfiguration'] = {'status': 'Disabled'}
        elif config.ssl_configuration and config.ssl_configuration.renew:
            if not json_payload['properties']['properties']['sslConfiguration'] or \
               not json_payload['properties']['properties']['sslConfiguration']['leafDomainLabel']:
                raise ComputeTargetException('Invalid configuration. When renew is set '
                                             'the cluster should have the leaf domain label set for SSL.')

            # Retain the configuration and update the renew flag
            json_payload['properties']['properties']['sslConfiguration']['renew'] = True
        elif config.ssl_configuration and config.ssl_configuration.cname:
            json_payload['properties']['properties']['sslConfiguration'] = {}
            json_payload['properties']['properties']['sslConfiguration']['status'] = 'Enabled'
            json_payload['properties']['properties']['sslConfiguration']['cname'] = config.ssl_configuration.cname
            json_payload['properties']['properties']['sslConfiguration']['cert'] = config.ssl_configuration.cert
            json_payload['properties']['properties']['sslConfiguration']['key'] = config.ssl_configuration.key
        elif config.ssl_configuration and config.ssl_configuration.leaf_domain_label:
            json_payload['properties']['properties']['sslConfiguration'] = {}
            json_payload['properties']['properties']['sslConfiguration']['status'] = 'Auto'
            json_payload['properties']['properties']['sslConfiguration']['leafDomainLabel'] = \
                config.ssl_configuration.leaf_domain_label
            json_payload['properties']['properties']['sslConfiguration']['overwriteExistingDomain'] = \
                config.ssl_configuration.overwrite_existing_domain or False
        else:
            del(json_payload['properties']['properties']['sslConfiguration'])

        if config.load_balancer_type:
            json_payload['properties']['properties']['loadBalancerType'] = config.load_balancer_type
        else:
            del(json_payload['properties']['properties']['loadBalancerType'])

        if config.load_balancer_subnet:
            json_payload['properties']['properties']['loadBalancerSubnet'] = config.load_balancer_subnet
        else:
            del(json_payload['properties']['properties']['loadBalancerSubnet'])

        return json_payload

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = AksCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location
        self.agent_vm_size = cluster.agent_vm_size
        self.agent_count = cluster.agent_count
        self.cluster_purpose = cluster.cluster_purpose
        self.cluster_fqdn = cluster.cluster_fqdn
        self.system_services = cluster.system_services
        self.ssl_configuration = cluster.ssl_configuration
        self.load_balancer_type = cluster.load_balancer_type
        self.load_balancer_subnet = cluster.load_balancer_subnet

    def delete(self):
        """Remove the AksCompute object from its associated workspace.

        If this object was created through Azure Machine Learning, the corresponding cloud-based objects
        will also be deleted. If this object was created externally and only attached to the workspace,
        this method raises a :class:`azureml.exceptions.ComputeTargetException` and nothing is changed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('delete')

    def detach(self):
        """Detach the AksCompute object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises azureml.exceptions.ComputeTargetException:
        """
        self._delete_or_detach('detach')

    def update(self, update_configuration):
        """Update the AksCompute object using the update configuration provided.

        :param update_configuration: An AKS update configuration object.
        :type update_configuration: azureml.core.compute.aks.AksUpdateConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        update_configuration.validate_configuration()
        update_payload = self._build_update_payload(update_configuration)

        # Update the compute by calling PUT on the existing compute
        updated_target = ComputeTarget._create_compute_target(
            self.workspace,
            self.name,
            update_payload,
            AksCompute)

        # This is useful while waiting for the operation to complete
        self._operation_endpoint = updated_target._operation_endpoint

    def get_credentials(self):
        """Retrieve the credentials for the AKS target.

        :return: The credentials for the AKS target.
        :rtype: dict
        :raises azureml.exceptions.ComputeTargetException:
        """
        endpoint = self._mlc_endpoint + '/listKeys'
        headers = self._auth.get_authentication_header()
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
        creds_content = json.loads(content)
        return creds_content

    def serialize(self):
        """Convert this AksCompute object into a json serialized dictionary.

        :return: The JSON representation of this AksCompute object.
        :rtype: dict
        """
        system_services = [system_service.serialize() for system_service in self.system_services] \
            if self.system_services else None

        ssl_configuration = self.ssl_configuration.serialize() if self.ssl_configuration else None

        aks_properties = {'agentVmSize': self.agent_vm_size, 'agentCount': self.agent_count,
                          'clusterPurpose': self.cluster_purpose, 'clusterFqdn': self.cluster_fqdn,
                          'systemServices': system_services, 'sslConfiguration': ssl_configuration,
                          'loadBalancerType': self.load_balancer_type,
                          'loadBalancerSubnet': self.load_balancer_subnet}

        cluster_properties = {'computeType': self.type, 'computeLocation': self.cluster_location,
                              'description': self.description, 'resourceId': self.cluster_resource_id,
                              'isAttachedCompute': self.is_attached,
                              'provisioningState': self.provisioning_state,
                              'provisioningErrors': self.provisioning_errors, 'properties': aks_properties}

        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': cluster_properties}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into an AksCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the AksCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to an AksCompute object.
        :type object_dict: dict
        :return: The AksCompute representation of the provided JSON object.
        :rtype: azureml.core.compute.aks.AksCompute
        :raises azureml.exceptions.ComputeTargetException:
        """
        AksCompute._validate_get_payload(object_dict)
        target = AksCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        """Validate get payload.

        :param payload:
        :type payload: dict
        :return:
        :rtype: None
        """
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != AksCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(AksCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))
        aks_properties = payload['properties']['properties']
        if aks_properties:
            for aks_key in ['agentVmSize', 'agentCount', 'clusterPurpose', 'clusterFqdn', 'systemServices']:
                if aks_key not in aks_properties:
                    raise ComputeTargetException('Invalid cluster payload, missing '
                                                 '["properties"]["properties"]["{}"]:\n'
                                                 '{}'.format(aks_key, payload))

    class ClusterPurpose(object):
        """Defines constants describing how an Azure Kubernetes Service cluster is used.

        These constants are used when provisioning Azure Machine Learning components to ensure
        the desired level of fault-tolerance and QoS.

        * 'FAST_PROD' will provision components to handle higher levels of traffic
          with production quality fault-tolerance. This will default the AKS cluster to
          have 3 nodes.

        * 'DEV_TEST' will provision components at a minimal level for testing.
          This will default the AKS cluster to have 1 node.

        Specify ClusterPurpose in the :meth:`azureml.core.compute.aks.AksCompute.attach_configuration` method of
        the :class:`azureml.core.compute.aks.AksCompute` class.

        For more information about these constants, see `Attach an existing AKS
        cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.
        """

        FAST_PROD = "FastProd"
        DEV_TEST = "DevTest"


class AksProvisioningConfiguration(ComputeTargetProvisioningConfiguration):
    """Represents configuration parameters for provisioning AksCompute targets.

    Use the ``provisioning_configuration`` method of the
    :class:`azureml.core.compute.aks.AksCompute` class to
    specify provisioning parameters.

    :param agent_count: The number of agents (VMs) to host containers. Defaults to 3.
    :type agent_count: int
    :param vm_size: The size of agent VMs. A full list of options can be found here:
        https://aka.ms/azureml-aks-details. Defaults to Standard_D3_v2.
    :type vm_size: str
    :param ssl_cname: A CNAME to use if enabling SSL validation on the cluster. Must provide all three
        CName, cert file, and key file to enable SSL validation
    :type ssl_cname: str
    :param ssl_cert_pem_file: A file path to a file containing cert information for SSL validation. Must provide
        all three CName, cert file, and key file to enable SSL validation
    :type ssl_cert_pem_file: str
    :param ssl_key_pem_file: A file path to a file containing key information for SSL validation. Must provide
        all three CName, cert file, and key file to enable SSL validation
    :type ssl_key_pem_file: str
    :param location: The location to provision cluster in. If not specified, will default to workspace location.
        Available regions for this compute can be found here:
        https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=kubernetes-service
    :type location: str
    :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located
    :type vnet_resourcegroup_name: str
    :param vnet_name: The name of the virtual network.
    :type vnet_name: str
    :param subnet_name: The name of the subnet inside the vnet
    :type subnet_name: str
    :param service_cidr: An IP range, in CIDR notation, from which to assign service cluster IPs.
    :type service_cidr: str
    :param dns_service_ip: Containers DNS server IP address.
    :type dns_service_ip: str
    :param docker_bridge_cidr: A CIDR notation IP for Docker bridge.
    :type docker_bridge_cidr: str
    :param cluster_purpose: The targeted usage of the cluster. This is used to provision Azure Machine Learning
        components to ensure the desired level of fault-tolerance and QoS. The
        :class:`azureml.core.compute.aks.AksCompute.ClusterPurpose` class is provided for convenience
        to specify possible values. For more information, see `Attach an existing AKS
        cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.
    :type cluster_purpose: str
    :param load_balancer_type: Load balancer type of AKS cluster.
        Valid values are PublicIp and InternalLoadBalancer. Default value is PublicIp.
    :type load_balancer_type: str
    :param load_balancer_subnet: Load balancer subnet of AKS cluster.
        It can be used only when Internal Load Balancer is used as load balancer type. Default value is aks-subnet.
    :type load_balancer_subnet: str
    """

    def __init__(self, agent_count, vm_size, ssl_cname, ssl_cert_pem_file, ssl_key_pem_file, location,
                 vnet_resourcegroup_name, vnet_name, subnet_name, service_cidr, dns_service_ip,
                 docker_bridge_cidr, cluster_purpose, load_balancer_type, load_balancer_subnet):
        """Initialize a configuration object for provisioning an AKS compute target.

        Must provide all three CName, cert file, and key file to enable SSL validation.

        :param agent_count: The number of agents (VMs) to host containers. Defaults to 3.
        :type agent_count: int
        :param vm_size: The size of agent VMs. A full list of options can be found here:
            https://aka.ms/azureml-aks-details. Defaults to Standard_D3_v2.
        :type vm_size: str
        :param ssl_cname: A CNAME to use if enabling SSL validation on the cluster. Must provide all three
            CName, cert file, and key file to enable SSL validation
        :type ssl_cname: str
        :param ssl_cert_pem_file: A file path to a file containing cert information for SSL validation. Must provide
            all three CName, cert file, and key file to enable SSL validation
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: A file path to a file containing key information for SSL validation. Must provide
            all three CName, cert file, and key file to enable SSL validation
        :type ssl_key_pem_file: str
        :param location: The location to provision cluster in. If not specified, will default to workspace location.
            Available regions for this compute can be found here:
            https://azure.microsoft.com/global-infrastructure/services/?regions=all&products=kubernetes-service
        :type location: str
        :param vnet_resourcegroup_name: The name of the resource group where the virtual network is located
        :type vnet_resourcegroup_name: str
        :param vnet_name: The name of the virtual network.
        :type vnet_name: str
        :param subnet_name: The name of the subnet inside the vnet
        :type subnet_name: str
        :param service_cidr: An IP range, in CIDR notation, from which to assign service cluster IPs.
        :type service_cidr: str
        :param dns_service_ip: Containers DNS server IP address.
        :type dns_service_ip: str
        :param docker_bridge_cidr: A CIDR notation IP for Docker bridge.
        :type docker_bridge_cidr: str
        :param cluster_purpose: The targeted usage of the cluster. This is used to provision Azure Machine Learning
          components to ensure the desired level of fault-tolerance and QoS. The
          :class:`azureml.core.compute.aks.AksCompute.ClusterPurpose` class is provided for convenience to
          specify possible values. For more information, see `Attach an existing AKS
          cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.
        :type cluster_purpose: str
        :param load_balancer_type: Load balancer type of AKS cluster.
            Valid values are PublicIp and InternalLoadBalancer. Default value is PublicIp.
        :type load_balancer_type: str
        :param load_balancer_subnet: Load balancer subnet of AKS cluster.
            It can be used only when Internal Load Balancer is used as load balancer type. Default value is aks-subnet.
        :type load_balancer_subnet: str
        :return: A configuration object to be used when creating a Compute object
        :rtype: azureml.core.compute.ask.AksProvisioningConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        super(AksProvisioningConfiguration, self).__init__(AksCompute, location)
        self.agent_count = agent_count
        self.vm_size = vm_size
        self.ssl_cname = ssl_cname
        self.ssl_cert_pem_file = ssl_cert_pem_file
        self.ssl_key_pem_file = ssl_key_pem_file
        self.vnet_resourcegroup_name = vnet_resourcegroup_name
        self.vnet_name = vnet_name
        self.subnet_name = subnet_name
        self.leaf_domain_label = None
        self.overwrite_existing_domain = False
        self.service_cidr = service_cidr
        self.dns_service_ip = dns_service_ip
        self.docker_bridge_cidr = docker_bridge_cidr
        self.cluster_purpose = cluster_purpose
        self.load_balancer_type = load_balancer_type
        self.load_balancer_subnet = load_balancer_subnet
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.

        :raises azureml.exceptions.ComputeTargetException:
        """
        if self.agent_count and self.agent_count <= 0:
            raise ComputeTargetException('Invalid configuration, agent count must be a positive integer.')
        if self.ssl_cname or self.ssl_cert_pem_file or self.ssl_key_pem_file:
            if not self.ssl_cname or not self.ssl_cert_pem_file or not self.ssl_key_pem_file:
                raise ComputeTargetException('Invalid configuration, not all ssl information provided. To enable SSL '
                                             'validation please provide the cname, cert pem file, and key pem file.')
            if self.leaf_domain_label:
                raise ComputeTargetException('Invalid configuration. When cname, cert pem file, and key pem file is '
                                             'provided, leaf domain label should not be provided.')
            if self.overwrite_existing_domain:
                raise ComputeTargetException('Invalid configuration. Overwrite existing domain only applies to leaf '
                                             'domain label. When cname, cert pem file, and key pem file is provided, '
                                             'Overwrite existing domain should not be provided.')
        elif self.leaf_domain_label:
            if self.ssl_cname or self.ssl_cert_pem_file or self.ssl_key_pem_file:
                raise ComputeTargetException('Invalid configuration. When leaf domain label is provided, cname, cert '
                                             'pem file, or key pem file should not be provided.')
        if self.vnet_name or self.vnet_resourcegroup_name or self.subnet_name or self.service_cidr or \
                self.dns_service_ip or self.docker_bridge_cidr:
            if not self.vnet_name or not self.vnet_resourcegroup_name or not self.subnet_name or \
                    not self.service_cidr or not self.dns_service_ip or not self.docker_bridge_cidr:
                raise ComputeTargetException('Invalid configuration, not all virtual net information provided. To use '
                                             'a custom virtual net with aks, please provide vnet name, vnet resource '
                                             'group, subnet name, service cidr, dns service ip and docker bridge cidr')

    def enable_ssl(self, ssl_cname=None, ssl_cert_pem_file=None, ssl_key_pem_file=None, leaf_domain_label=None,
                   overwrite_existing_domain=False):
        """Enable SSL validation on the cluster.

        :param ssl_cname: A CNAME to use if enabling SSL validation on the cluster.
            To enable SSL validation, you must provide the three related parameters:
            CNAME, cert PEM file, and key PEM file.
        :type ssl_cname: str
        :param ssl_cert_pem_file: A file path to a file containing cert information for SSL validation.
            To enable SSL validation, you must provide the three related parameters:
            CNAME, cert PEM file, and key PEM file.
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: A file path to a file containing key information for SSL validation.
            To enable SSL validation, you must provide the three related parameters:
            CNAME, cert PEM file, and key PEM file.
        :type ssl_key_pem_file: str
        :param leaf_domain_label: The leaf domain label to use if enabling SSL validation on the cluster.
            When leaf domain label is provided, do not specify CNAME, cert PEM file, or key PEM file.
        :type leaf_domain_label: str
        :param overwrite_existing_domain: Whether or not to overwrite the existing leaf domain label. Overwrite of an
            existing domain only applies to leaf domain label. When this parameter is provided,
            CNAME, cert PEM file, and key PEM file should not be provided.
        :type overwrite_existing_domain: bool
        """
        if ssl_cname:
            self.ssl_cname = ssl_cname
        if ssl_cert_pem_file:
            self.ssl_cert_pem_file = ssl_cert_pem_file
        if ssl_key_pem_file:
            self.ssl_key_pem_file = ssl_key_pem_file
        if leaf_domain_label:
            self.leaf_domain_label = leaf_domain_label
        if overwrite_existing_domain:
            self.overwrite_existing_domain = overwrite_existing_domain
        self.validate_configuration()
        return self


class AksAttachConfiguration(ComputeTargetAttachConfiguration):
    """Represents configuration parameters for attaching AksCompute targets.

    Use the ``attach_configuration`` method of the
    :class:`azureml.core.compute.aks.AksCompute` class to
    specify attach parameters.

    :param resource_group: The name of the resource group in which the AKS cluster is located.
    :type resource_group: str
    :param cluster_name: The AKS cluster name.
    :type cluster_name: str
    :param resource_id: The Azure resource ID for the compute resource being attached.
    :type resource_id: str
    :param cluster_purpose: The targeted usage of the cluster. This is used to provision Azure Machine
        Learning components to ensure the desired level of fault-tolerance and QoS. The
        :class:`azureml.core.compute.aks.AksCompute.ClusterPurpose` class defines the possible values.
        For more information, see `Attach an existing AKS
        cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.
    :type cluster_purpose: str
    """

    def __init__(
            self, resource_group=None, cluster_name=None, resource_id=None, cluster_purpose=None,
            load_balancer_type=None, load_balancer_subnet=None
    ):
        """Initialize the configuration object.

        :param resource_group: The name of the resource group in which the AKS cluster is located.
        :type resource_group: str
        :param cluster_name: The AKS cluster name.
        :type cluster_name: str
        :param resource_id: The Azure resource ID for the compute resource being attached.
        :type resource_id: str
        :param cluster_purpose: The targeted usage of the cluster. This is used to provision Azure Machine
          Learning components to ensure the desired level of fault-tolerance and QoS. The
          :class:`azureml.core.compute.aks.AksCompute.ClusterPurpose` class defines the possible values.
          For more information, see `Attach an existing AKS
          cluster <https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service>`_.
        :type cluster_purpose: str
        :param load_balancer_type: The AKS cluster type. Valid values are PublicIp and InternalLoadBalancer.
          Default value is PublicIp.
        :type load_balancer_type: str
        :param load_balancer_subnet: The AKS load balancer subnet. It can be used only when InternalLoadBalancer
          is used as load balancer type. Default value is aks-subnet.
        :type load_balancer_subnet: str
        :return: The configuration object
        :rtype: azureml.core.compute.aks.AksAttachConfiguration
        """
        super(AksAttachConfiguration, self).__init__(AksCompute)
        self.resource_group = resource_group
        self.cluster_name = cluster_name
        self.resource_id = resource_id
        self.cluster_purpose = cluster_purpose
        self.ssl_cname = None
        self.ssl_cert_pem_file = None
        self.ssl_key_pem_file = None
        self.leaf_domain_label = None
        self.overwrite_existing_domain = False
        self.load_balancer_type = load_balancer_type
        self.load_balancer_subnet = load_balancer_subnet
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
            if resource_type != 'Microsoft.ContainerService':
                raise ComputeTargetException('Invalid resource_id provided, resource type {} does not match for '
                                             'AKS'.format(resource_type))
            # make sure do not use other info
            if self.resource_group:
                raise ComputeTargetException('Since resource_id is provided, please do not provide resource_group.')
            if self.cluster_name:
                raise ComputeTargetException('Since resource_id is provided, please do not provide cluster_name.')
        elif self.resource_group or self.cluster_name:
            # resource_id is not provided, validate other info
            if not self.resource_group:
                raise ComputeTargetException('resource_group is not provided.')
            if not self.cluster_name:
                raise ComputeTargetException('cluster_name is not provided.')
        else:
            # neither resource_id nor other info is provided
            raise ComputeTargetException('Please provide resource_group and cluster_name for the AKS compute '
                                         'resource being attached. Or please provide resource_id for the resource '
                                         'being attached.')

        if self.ssl_cname or self.ssl_cert_pem_file or self.ssl_key_pem_file:
            if not self.ssl_cname or not self.ssl_cert_pem_file or not self.ssl_key_pem_file:
                raise ComputeTargetException('Invalid configuration, not all ssl information provided. To enable SSL '
                                             'validation please provide the cname, cert pem file, and key pem file.')
            if self.leaf_domain_label:
                raise ComputeTargetException('Invalid configuration. When cname, cert pem file, and key pem file is '
                                             'provided, leaf domain label should not be provided.')
            if self.overwrite_existing_domain:
                raise ComputeTargetException('Invalid configuration. Overwrite existing domain only applies to leaf '
                                             'domain label. When cname, cert pem file, and key pem file is provided, '
                                             'Overwrite existing domain should not be provided.')
        elif self.leaf_domain_label:
            if self.ssl_cname or self.ssl_cert_pem_file or self.ssl_key_pem_file:
                raise ComputeTargetException('Invalid configuration. When leaf domain label is provided, cname, cert '
                                             'pem file, or key pem file should not be provided.')

        subnet_error_msg = 'Invalid configuration, load balancer subnet can be changed from default when load ' \
                           'balancer type is "InternalLoadBalancer".'
        if self.load_balancer_type:
            if self.load_balancer_type not in ("InternalLoadBalancer", "PublicIp"):
                raise ComputeTargetException('Invalid configuration. Load balancer type should be either "PublicIp" or'
                                             ' "InternalLoadBalancer".')
            if self.load_balancer_type == "PublicIp" and self.load_balancer_subnet:
                raise ComputeTargetException(subnet_error_msg)
        elif self.load_balancer_subnet:
            raise ComputeTargetException(subnet_error_msg)

    def enable_ssl(self, ssl_cname=None, ssl_cert_pem_file=None, ssl_key_pem_file=None, leaf_domain_label=None,
                   overwrite_existing_domain=False):
        """Enable SSL validation on the AKS cluster.

        :param ssl_cname: A CNAME to use if enabling SSL validation on the cluster.
            To enable SSL validation, you must provide the three related parameters:
            CNAME, cert PEM file, and key PEM file.
        :type ssl_cname: str
        :param ssl_cert_pem_file: A file path to a file containing cert information for SSL validation.
            To enable SSL validation, you must provide the three related parameters:
            CNAME, cert PEM file, and key PEM file.
        :type ssl_cert_pem_file: str
        :param ssl_key_pem_file: A file path to a file containing key information for SSL validation.
            To enable SSL validation, you must provide the three related parameters:
            CNAME, cert PEM file, and key PEM file.
        :type ssl_key_pem_file: str
        :param leaf_domain_label: The leaf domain label to use if enabling SSL validation on the cluster.
            When leaf domain label is provided, do not specify CNAME, cert PEM file, or key PEM file.
        :type leaf_domain_label: str
        :param overwrite_existing_domain: Whether to overwrite the existing leaf domain label. Overwrite of an
            existing domain only applies to leaf domain label. When this parameter is provided,
            CNAME, cert PEM file, and key PEM file should not be provided.
        :type overwrite_existing_domain: bool
        """
        if ssl_cname:
            self.ssl_cname = ssl_cname
        if ssl_cert_pem_file:
            self.ssl_cert_pem_file = ssl_cert_pem_file
        if ssl_key_pem_file:
            self.ssl_key_pem_file = ssl_key_pem_file
        if leaf_domain_label:
            self.leaf_domain_label = leaf_domain_label
        if overwrite_existing_domain:
            self.overwrite_existing_domain = overwrite_existing_domain
        self.validate_configuration()
        return self


class AksUpdateConfiguration(ComputeTargetUpdateConfiguration):
    """Represents configuration parameters for updating an AKSCompute target.

    :param ssl_configuration: SSL configuration parameters.
    :type ssl_configuration: azureml.core.compute.aks.SslConfiguration
    :param load_balancer_type: Load balancer type of AKS cluster.
        Valid values are PublicIp and InternalLoadBalancer. Default value is PublicIp.
    :type load_balancer_type: str
    :param load_balancer_subnet: Load balancer subnet of AKS cluster.
        It can be used only when Internal Load Balancer is used as load balancer type. Default value is aks-subnet.
    :type load_balancer_subnet: str
    """

    def __init__(self, ssl_configuration=None, load_balancer_type=None, load_balancer_subnet=None):
        """Initialize the configuration object.

        :return: The configuration object
        :rtype: azureml.core.compute.aks.AksUpdateConfiguration
        """
        super(AksUpdateConfiguration, self).__init__(AksCompute)
        self.ssl_configuration = ssl_configuration
        self.load_balancer_type = load_balancer_type
        self.load_balancer_subnet = load_balancer_subnet
        self.validate_configuration()

    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        Raises a :class:`azureml.exceptions.ComputeTargetException` if validation fails.
        :raises azureml.exceptions.ComputeTargetException:
        """
        if self.ssl_configuration:
            if self.ssl_configuration.cname or self.ssl_configuration.cert or \
               self.ssl_configuration.key:
                if not self.ssl_configuration.cname or not self.ssl_configuration.cert or \
                   not self.ssl_configuration.key:
                    raise ComputeTargetException('Invalid configuration, not all ssl information provided. '
                                                 'To enable SSL validation please provide the cname, cert '
                                                 'pem file, and key pem file.')
                if self.ssl_configuration.leaf_domain_label:
                    raise ComputeTargetException('Invalid configuration. When cname, cert pem file, '
                                                 'and key pem file is provided, leaf domain label should '
                                                 'not be provided.')
                if self.ssl_configuration.overwrite_existing_domain:
                    raise ComputeTargetException('Invalid configuration. Overwrite existing domain only '
                                                 'applies to leaf domain label. When cname, cert pem file, '
                                                 'and key pem file is provided, Overwrite existing domain '
                                                 'should not be provided.')
            elif self.ssl_configuration.leaf_domain_label:
                if self.ssl_configuration.cname or self.ssl_configuration.cert or self.ssl_configuration.key:
                    raise ComputeTargetException('Invalid configuration. When leaf domain label is provided, '
                                                 'cname, cert pem file, or key pem file should not be provided.')

        if self.load_balancer_subnet and self.load_balancer_type == 'PublicIp':
            raise ComputeTargetException('Invalid configuration, load balancer subnet '
                                         'cannot be used with a Public IP.')


class SystemService(object):
    """Represents an Azure Kubernetes System Service object.

        For more information about services in Azure Kubernetes, see
        `Network concepts for applications in AKS <https://docs.microsoft.com/azure/aks/concepts-network#services>`_.

    :param service_type: The underlying type associated with this service.
    :type service_type: str
    :param version: The service version.
    :type version: str
    :param public_ip_address: The accessible IP address for this service.
    :type public_ip_address: str
    """

    def __init__(self, service_type, version, public_ip_address):
        """Initialize the System Service object.

        :param service_type: The underlying type associated with this service.
        :type service_type: str
        :param version: The service version.
        :type version: str
        :param public_ip_address: The accessible IP address for this service.
        :type public_ip_address: str
        """
        self.service_type = service_type
        self.version = version
        self.public_ip_address = public_ip_address

    def serialize(self):
        """Convert this SystemService object into a JSON serialized dictionary.

        :return: The JSON representation of this SystemService object.
        :rtype: dict
        """
        return {'serviceType': self.service_type, 'version': self.version, 'publicIpAddress': self.public_ip_address}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into a SystemService object.

        :param object_dict: A JSON object to convert to a SystemService object.
        :type object_dict: dict
        :return: The SystemService representation of the provided JSON object.
        :rtype: azureml.core.compute.aks.SystemService
        :raises azureml.exceptions.ComputeTargetException:
        """
        for service_key in ['systemServiceType', 'version', 'publicIpAddress']:
            if service_key not in object_dict:
                raise ComputeTargetException('Invalid system service payload, missing "{}":\n'
                                             '{}'.format(service_key, object_dict))
        return SystemService(object_dict['systemServiceType'], object_dict['version'], object_dict['publicIpAddress'])


class SslConfiguration(object):
    """Represents an SSL configuration object for use with AksCompute.

    .. remarks::

        To configure SSL, specify either the ``leaf_domain_label`` parameter or the parameters ``cname``,
        ``cert``, and ``key``.

        A typical pattern to specify SSL configuration is to use the ``attach_configuration`` or
        ``provisioning_configuration`` method of the
        :class:`azureml.core.compute.aks.AksCompute` class to obtain a configuration object. Then, use the
        ``enable_ssl`` method of the returned configuration object. For example, for the attach configuration,
        use the :meth:`azureml.core.compute.aks.AksAttachConfiguration.enable_ssl` method.

        .. code-block:: python

            # Load workspace configuration from the config.json file.
            from azureml.core import Workspace
            ws = Workspace.from_config()

            # Use the default configuration, but you can also provide parameters to customize.
            from azureml.core.compute import AksCompute
            prov_config = AksCompute.provisioning_configuration()
            attach_config = AksCompute.attach_configuration(resource_group=ws.resource_group,
                                                            cluster_name="dev-cluster")

            # Enable ssl.
            prov_config.enable_ssl(leaf_domain_label = "contoso")
            attach_config.enable_ssl(leaf_domain_label = "contoso")

        For more information on enabling SSL for AKS, see `Use SSL to secure a web service through Azure Machine
        Learning <https://docs.microsoft.com/azure/machine-learning/how-to-secure-web-service#enable>`_.

    :param status: Indicates whether SSL validation is enabled, disabled, or auto.
    :type status: str
    :param cert: The cert string to use for SSL validation. If provided, you must also provide ``cname`` and
        ``key`` PEM file
    :type cert: str
    :param key: The key string to use for SSL validation. If provided, you must also provide ``cname`` and
        ``cert`` PEM file
    :type key: str
    :param cname: The CNAME to use for SSL validation. If provided, you must also provide ``cert`` and ``key``
        PEM files.
    :type cname: str
    :param leaf_domain_label: The leaf domain label to use for the auto-generated certificate.
    :type leaf_domain_label: str
    :param overwrite_existing_domain: Indicates whether to overwrite the existing leaf domain label. The default
        is False.
    :type overwrite_existing_domain: bool
    :param renew: Indicates if ``leaf_domain_label`` refreshes the auto-generated certificate. If provided,
        the existing SSL configuration must be auto. The default is False.
    :type renew: bool
    """

    def __init__(self, status=None, cert=None, key=None, cname=None, leaf_domain_label=None,
                 overwrite_existing_domain=False, renew=False):
        """Initialize the SslConfiguration object.

        :param status: Indicates whether SSL validation is enabled, disabled, or auto.
        :type status: str
        :param cert: The cert string to use for SSL validation. If provided, you must also provide ``cname`` and
            ``key`` PEM file
        :type cert: str
        :param key: The key string to use for SSL validation. If provided, you must also provide ``cname`` and
            ``cert`` PEM file
        :type key: str
        :param cname: The CNAME to use for SSL validation. If provided, you must also provide ``cert`` and ``key``
            PEM files.
        :type cname: str
        :param leaf_domain_label: The leaf domain label to use for the auto-generated certificate.
        :type leaf_domain_label: str
        :param overwrite_existing_domain: Indicates whether to overwrite the existing leaf domain label. The default
            is False.
        :type overwrite_existing_domain: bool
        :param renew: Indicates if ``leaf_domain_label`` refreshes the auto-generated certificate. If provided,
            the existing SSL configuration must be auto. The default is False.
        :type renew: bool
        """
        self.status = status
        self.cert = cert
        self.key = key
        self.cname = cname
        self.leaf_domain_label = leaf_domain_label
        self.overwrite_existing_domain = overwrite_existing_domain
        self.renew = renew

    def serialize(self):
        """Convert this SslConfiguration object into a JSON serialized dictionary.

        :return: The JSON representation of this SslConfiguration object.
        :rtype: dict
        """
        return {
            'status': self.status, 'cert': self.cert, 'key': self.key, 'cname': self.cname,
            'leafDomainLabel': self.leaf_domain_label, 'overwriteExistingDomain': self.overwrite_existing_domain,
            'renew': self.renew}

    @staticmethod
    def deserialize(object_dict):
        """Convert a JSON object into a SslConfiguration object.

        :param object_dict: A JSON object to convert to a SslConfiguration object.
        :type object_dict: dict
        :return: The SslConfiguration representation of the provided JSON object.
        :rtype: azureml.core.compute.aks.SslConfiguration
        :raises azureml.exceptions.ComputeTargetException:
        """
        status = object_dict.get('status', None)
        cert = object_dict.get('cert', None)
        key = object_dict.get('key', None)
        cname = object_dict.get('cname', None)
        leaf_domain_label = object_dict.get('leafDomainLabel', None)
        overwrite_existing_domain = object_dict.get('overwriteExistingDomain', False)
        renew = object_dict.get('renew', False)
        return SslConfiguration(status, cert, key, cname, leaf_domain_label, overwrite_existing_domain, renew)
