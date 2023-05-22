# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from collections import OrderedDict
import json
import uuid


class ArmTemplateBuilder(object):

    def __init__(self):
        template = OrderedDict()
        template['$schema'] = \
            'https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#'
        template['contentVersion'] = '1.0.0.0'
        template['parameters'] = {}
        template['variables'] = {}
        template['resources'] = []
        self.template = template

    def add_resource(self, resource):
        self.template['resources'].append(resource)

    def build(self):
        return json.loads(json.dumps(self.template, default=set_default))


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def build_storage_account_resource(name, location, tags):
    storage_account = {
        'type': 'Microsoft.Storage/storageAccounts',
        'name': name,
        'apiVersion': '2018-07-01',
        'location': location,
        'sku': {
            'name': 'Standard_LRS'
        },
        'kind': 'StorageV2',
        'tags': tags,
        'dependsOn': [],
        'properties': {
            "encryption": {
                "services": {
                    "blob": {
                        "enabled": 'true'
                    },
                    "file": {
                        "enabled": 'true'
                    }
                },
                "keySource": "Microsoft.Storage"
            },
            "supportsHttpsTrafficOnly": True,
            "allowBlobPublicAccess": False
        }
    }
    return storage_account


def build_application_insights_resource(name, location, tags, default_log_analytics_arm_id):
    application_insights = {
        'type': 'microsoft.insights/components',
        'name': name,
        'kind': 'web',
        'apiVersion': '2020-02-02-preview',
        'location': location,
        'tags': tags,
        'properties': {
            'Application_Type': 'web',
            'WorkspaceResourceId': default_log_analytics_arm_id
        }
    }
    return application_insights


def build_default_resource_group_deployment(deployment_name, location, subscription_id):
    resource_group = {
        "type": "Microsoft.Resources/deployments",
        "apiVersion": "2019-10-01",
        "name": deployment_name,
        "subscriptionId": subscription_id,
        "location": location,
        "properties": {
            "mode": "Incremental",
            "template": {
                "$schema": "https://schema.management.azure.com/schemas/\
                    2018-05-01/subscriptionDeploymentTemplate.json#",
                "contentVersion": "1.0.0.1",
                "parameters": {},
                "variables": {},
                "resources": [
                    {
                        "type": "Microsoft.Resources/resourceGroups",
                        "apiVersion": "2018-05-01",
                        "location": location,
                        "name": f"DefaultResourceGroup-{location}",
                        "properties": {}
                    }]
            }
        }
    }
    return resource_group


def build_default_log_analytics_deployment(deployment_name, location, subscription_id):
    log_analytics = {
        "type": "Microsoft.Resources/deployments",
        "apiVersion": "2019-10-01",
        "name": deployment_name,
        "resourceGroup": f"DefaultResourceGroup-{location}",
        "subscriptionId": subscription_id,
        "properties": {
            "mode": "Incremental",
            "template": {
                "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "parameters": {},
                "variables": {},
                "resources": [
                    {
                        "apiVersion": "2020-08-01",
                        "name": f"DefaultWorkspace-{location}",
                        "type": "Microsoft.OperationalInsights/workspaces",
                        "location": location,
                        "properties": {}
                    }]
            }
        }
    }
    return log_analytics


def build_keyvault_account_resource(name, location, tenantId, tags):
    keyvault_account = {
        'type': 'Microsoft.KeyVault/vaults',
        'name': name,
        'apiVersion': '2015-06-01',
        'location': location,
        'tags': tags,
        'dependsOn': [],
        'properties': {
                'enabledForDeployment': 'true',
                'enabledForTemplateDeployment': 'true',
                'enabledForVolumeEncryption': 'true',
                'tenantId': tenantId,
                'accessPolicies': [],
                'sku': {
                    'name': 'Standard',
                    'family': 'A'
                }
        }
    }
    return keyvault_account


def build_private_endpoint_resource(
        private_endpoint_configuration,
        location,
        workspace_resource_id,
        groupIds,
        private_endpoint_auto_approval,
        tags):
    private_endpoint = {
        "apiVersion": "2019-04-01",
        "name": private_endpoint_configuration.name,
        "type": "Microsoft.Network/privateEndpoints",
        "location": location,
        'tags': tags,
        "properties": {
            "subnet": {
                # Arm id of the subnet.
                # Example: /subscriptions/e54229a3-0e6f-40b3-82a1-ae9cda6e2b81/resourceGroups/suba-ws-vnet/providers/
                # Microsoft.Network/virtualNetworks/vnet-subaplcan/subnets/default"
                "id": "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network/virtualNetworks/{}"
                      "/subnets/{}".format(private_endpoint_configuration.vnet_subscription_id,
                                           private_endpoint_configuration.vnet_resource_group,
                                           private_endpoint_configuration.vnet_name,
                                           private_endpoint_configuration.vnet_subnet_name)
            }
        }
    }

    private_link_service_connection = {
        "name": private_endpoint_configuration.name,
        "properties": {
            "privateLinkServiceId": workspace_resource_id,
            "groupIds": groupIds
        }
    }
    if private_endpoint_auto_approval:
        private_endpoint["properties"]["privateLinkServiceConnections"] = [
            private_link_service_connection]
    else:
        private_endpoint["properties"]["manualPrivateLinkServiceConnections"] = [
            private_link_service_connection]

    return private_endpoint


def build_private_dns_zones(
        private_dns_deployment_name,
        private_endpoint_name,
        subscriptionId,
        resourceGroup,
        private_dns_zone_names,
        tags):
    private_dns_zone_resources = []
    for private_dns_zone_name in private_dns_zone_names:
        private_dns_zone_resources.append({
            "apiVersion": "2017-05-10",
            "name": "PrivateDnsZone-{}".format(str(uuid.uuid4())),
            "type": "Microsoft.Resources/deployments",
            "subscriptionId": subscriptionId,
            "resourceGroup": resourceGroup,
            "properties": {
                "mode": "Incremental",
                "template": {
                    "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                    "contentVersion": "1.0.0.0",
                    "resources": [
                        {
                            "type": "Microsoft.Network/privateDnsZones",
                            "apiVersion": "2018-09-01",
                            "name": private_dns_zone_name,
                            "location": "global",
                            "tags": tags,
                            "properties": {}
                        }
                    ]
                }
            }
        })
    return {
        "type": "Microsoft.Resources/deployments",
        "apiVersion": "2017-05-10",
        "name": private_dns_deployment_name,
        "dependsOn": [private_endpoint_name],
        "properties": {
            "mode": "Incremental",
            "template": {
                "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "resources": private_dns_zone_resources}}}


def build_virtual_network_links(
        private_dns_deployment_name,
        private_endpoint_name,
        virtual_network_id,
        subscriptionId,
        resourceGroup,
        private_dns_zone_names,
        tags):
    virtual_network_links = []
    for private_dns_zone_name in private_dns_zone_names:
        virtual_network_links.append({
            "apiVersion": "2017-05-10",
            "name": "VirtualNetworklink-{}".format(str(uuid.uuid4())),
            "type": "Microsoft.Resources/deployments",
            "subscriptionId": subscriptionId,
            "resourceGroup": resourceGroup,
            "properties": {
                "mode": "Incremental",
                "template": {
                    "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                    "contentVersion": "1.0.0.0",
                    "resources": [
                        {
                            "type": "Microsoft.Network/privateDnsZones/virtualNetworkLinks",
                            "apiVersion": "2018-09-01",
                            "name": "[concat('{}', '/', uniqueString('{}'))]"
                            .format(private_dns_zone_name, virtual_network_id),
                            "location": "global",
                            'tags': tags,
                            "properties": {
                                "virtualNetwork": {
                                    "id": virtual_network_id
                                },
                                "registrationEnabled": False
                            }
                        }
                    ]
                }
            }
        })
    return {
        "type": "Microsoft.Resources/deployments",
        "apiVersion": "2017-05-10",
        "name": "VirtualNetworkLink-{}".format(str(uuid.uuid4())),
        "dependsOn": [
            private_dns_deployment_name
        ],
        "properties": {
            "mode": "Incremental",
            "template": {
                "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "resources": virtual_network_links
            }
        }
    }


def build_private_dns_zone_groups(
        private_dns_deployment_name,
        private_endpoint_name,
        private_endpoint_location,
        subscriptionId,
        resourceGroup,
        private_dns_zone_names):
    private_dns_zone_configs = []
    for dns_zone_name in private_dns_zone_names:
        private_dns_zone_configs.append(
            {
                "name": "{}".format(
                    dns_zone_name.replace(
                        ".",
                        "-")),
                "properties": {
                    "privateDnsZoneId": (
                        "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Network"
                        "/privateDnsZones/{}").format(
                            subscriptionId,
                            resourceGroup,
                        dns_zone_name)}})
    return {
        "apiVersion": "2017-05-10",
        "name": "DnsZoneGroup-{}".format(str(uuid.uuid4())),
        "type": "Microsoft.Resources/deployments",
        "resourceGroup": "{}".format(resourceGroup),
        "dependsOn": [
            private_endpoint_name,
            private_dns_deployment_name
        ],
        "properties": {
            "mode": "Incremental",
            "template": {
                "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "resources": [
                    {
                        "type": "Microsoft.Network/privateEndpoints/privateDnsZoneGroups",
                        "apiVersion": "2020-03-01",
                        "name": "{}/default".format(private_endpoint_name),
                        "location": private_endpoint_location,
                        "properties": {
                            "privateDnsZoneConfigs": private_dns_zone_configs
                        }
                    }
                ]
            }
        }
    }


'''
def build_private_dns_zone_resource(private_endpoint_config, vnet_resource_id, dns_zone_name):
    vnet_link_name = "[concat('{}', '/', uniqueString('{}'))]".format(dns_zone_name, vnet_resource_id)
    private_dns_guid = str(uuid.uuid4())
    private_endpoint_resource_id = "resourceId('Microsoft.Network/privateEndpoints', '{}')".format(
        private_endpoint_config.name)
    name = "PrivateDns-{}".format(private_dns_guid)
    return {
        "type": "Microsoft.Resources/deployments",
        "apiVersion": "2017-05-10",
        "name": "{}".format(name),
        "dependsOn": [
            "[{}]".format(private_endpoint_resource_id)
        ],
        "properties": {
            "mode": "Incremental",
            "template": {
                "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "resources": [
                    {
                        "type": "Microsoft.Network/privateDnsZones",
                        "apiVersion": "2018-09-01",
                        "name": dns_zone_name,
                        "location": "global",
                        "tags": {},
                        "properties": {}
                    },
                    {
                        "type": "Microsoft.Network/privateDnsZones/virtualNetworkLinks",
                        "apiVersion": "2018-09-01",
                        "name": vnet_link_name,
                        "location": "global",
                        "dependsOn": [
                            dns_zone_name
                        ],
                        "properties": {
                            "virtualNetwork": {
                                "id": vnet_resource_id
                            },
                            "registrationEnabled": "false"
                        }
                    },
                    {
                        "apiVersion": "2017-05-10",
                        "name": "[concat('EndpointDnsRecords-', '{}')]".format(private_dns_guid),
                        "type": "Microsoft.Resources/deployments",
                        "dependsOn": [
                            dns_zone_name
                        ],
                        "properties": {
                            "mode": "Incremental",
                            "templatelink": {
                                "contentVersion": "1.0.0.0",
                                "uri": "https://network.hosting.portal.azure.net/network/Content/4.13.392.925/"
                                       "DeploymentTemplates/PrivateDnsForPrivateEndpoint.json"
                            },
                            "parameters": {
                                "privateDnsName": {
                                    "value": dns_zone_name
                                },
                                "privateEndpointNicResourceId": {
                                    "value": "[reference({}).networkInterfaces[0].id]".format(
                                        private_endpoint_resource_id)
                                },
                                "nicRecordsTemplateUri": {
                                    "value": "https://network.hosting.portal.azure.net/network/Content/4.13.392.925/"
                                             "DeploymentTemplates/PrivateDnsForPrivateEndpointNic.json"
                                },
                                "ipConfigRecordsTemplateUri": {
                                    "value": "https://network.hosting.portal.azure.net/network/Content/4.13.392.925/"
                                             "DeploymentTemplates/PrivateDnsForPrivateEndpointIpConfig.json"
                                },
                                "uniqueId": {
                                    "value": private_dns_guid
                                },
                                "existingRecords": {
                                    "value": {}
                                }
                            }
                        }
                    }
                ]
            }
        },
        "resourceGroup": "{}".format(private_endpoint_config.vnet_resource_group)
    }
'''


def build_workspace_resource(
        name,
        location,
        keyVault,
        containerRegistry,
        adbWorkspace,
        storageAccount,
        appInsights,
        primary_user_assigned_identity,
        cmkKeyVault,
        resourceCmkUri,
        hbiWorkspace,
        sku,
        tags,
        userAssignedIdentityForCmkEncryption,
        systemDatastoresAuthMode,
        v1LegacyMode,
        api_version):
    status = "Disabled"
    if resourceCmkUri:
        status = "Enabled"

    identity_type = 'SystemAssigned'
    userAssignedIdentities = None

    # if primary_user_assigned_identity is passed, the identity_type has to be
    # UserAssigned
    if primary_user_assigned_identity:
        identity_type = 'UserAssigned'
        if userAssignedIdentities is None:
            userAssignedIdentities = {}
        userAssignedIdentities[primary_user_assigned_identity] = {}

    if userAssignedIdentityForCmkEncryption:
        if primary_user_assigned_identity:
            identity_type = "UserAssigned"
        else:
            identity_type = "SystemAssigned,UserAssigned"
        if userAssignedIdentities is None:
            userAssignedIdentities = {}
        userAssignedIdentities[userAssignedIdentityForCmkEncryption] = {}

    identity = {
        'type': identity_type,
        'userAssignedIdentities': userAssignedIdentities
    }

    workspace_resource = {
        'type': 'Microsoft.MachineLearningServices/workspaces',
        'name': name,
        'apiVersion': api_version,
        'identity': identity,
        'location': location,
        'resources': [],
        'dependsOn': [],
        'sku': {
            'tier': sku,
            'name': sku
        },
        'tags': tags,
        'properties': {
            'containerRegistry': containerRegistry,
            'adbWorkspace': adbWorkspace,
            'keyVault': keyVault,
            'applicationInsights': appInsights,
            'friendlyName': name,
            'storageAccount': storageAccount,
            'primaryUserAssignedIdentity': primary_user_assigned_identity,
            "encryption": {
                "status": status,
                "identity": {
                    "userAssignedIdentity": userAssignedIdentityForCmkEncryption
                },
                "keyVaultProperties": {
                    "keyVaultArmId": cmkKeyVault,
                    "keyIdentifier": resourceCmkUri,
                    "identityClientId": ""
                }
            },
            'hbiWorkspace': hbiWorkspace,
            'systemDatastoresAuthMode': systemDatastoresAuthMode,
            "v1LegacyMode": v1LegacyMode,
        }
    }
    return workspace_resource
