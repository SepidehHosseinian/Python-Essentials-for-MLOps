# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import time
import uuid

from ._utils import ArmDeploymentOrchestrator, get_arm_resource_id
from azureml._project import _commands
from .arm_template_builder import (
    ArmTemplateBuilder,
    build_private_endpoint_resource,
    build_private_dns_zone_groups,
    build_private_dns_zones,
    build_virtual_network_links
)

PRIVATE_DNS_ZONE = "PrivateDnsZone"
PRIVATE_END_POINT = "PrivateEndPoint"


class PrivateEndPointArmDeploymentOrchestrator(ArmDeploymentOrchestrator):

    def __init__(
            self,
            auth,
            workspace_resource_group_name,
            location,
            workspace_subscription_id,
            workspace_name,
            private_endpoint_config,
            private_endpoint_auto_approval,
            deployment_name=None,
            tags=None):

        import random

        super().__init__(
            auth,
            private_endpoint_config.vnet_resource_group,
            private_endpoint_config.vnet_subscription_id,
            deployment_name if deployment_name else '{0}_{1}'.format(
                'Microsoft.PrivateEndpoint',
                random.randint(
                    100,
                    99999)))
        self.auth = auth
        self.workspace_subscription_id = workspace_subscription_id
        self.workspace_resource_group_name = workspace_resource_group_name
        self.workspace_name = workspace_name
        self.master_template = ArmTemplateBuilder()
        self.location = location.lower().replace(" ", "")
        self.resources_being_deployed = {}
        self.tags = tags

        self.private_endpoint_config = private_endpoint_config
        self.private_endpoint_auto_approval = private_endpoint_auto_approval
        self._init_private_dns_zone_names_and_group_id()
        self.error = None

    def deploy_private_endpoint(self, show_output=True):
        try:
            self._generate_private_endpoint_resource()
            # build the template
            template = self.master_template.build()

            # deploy the template
            self._arm_deploy_template(template)

            if show_output:
                while not self.poller.done():
                    self._check_deployment_status()
                    time.sleep(5)

                if self.poller._exception is not None:
                    self.error = self.poller._exception
                else:
                    # one last check to make sure all print statements make it
                    self._check_deployment_status()
            else:
                try:
                    self.poller.wait()
                except Exception:
                    self.error = self.poller._exception
        except Exception as ex:
            self.error = ex

        if self.error is not None:
            error_msg = "Unable to create the private endpoint for the workspace. \n {}".format(
                self.error)
            print(error_msg)

    def _generate_private_endpoint_resource(self):
        # The vnet, subnet should be precreated by user
        workspace_resource_id = get_arm_resource_id(
            self.workspace_resource_group_name,
            "Microsoft.MachineLearningServices/workspaces",
            self.workspace_name,
            self.workspace_subscription_id)
        self.master_template.add_resource(
            build_private_endpoint_resource(
                self.private_endpoint_config,
                self.location,
                workspace_resource_id,
                self.group_ids,
                self.private_endpoint_auto_approval,
                self.tags))
        self.resources_being_deployed[self.private_endpoint_config.name] = (
            PRIVATE_END_POINT, None)

        if self.private_endpoint_auto_approval is not None and self.private_endpoint_auto_approval is True:
            # Create private dns zone only for auto approval.
            # For manual approval, users have to approve the PE and manually
            # create the private DNS zone.
            vnet_resource_id = get_arm_resource_id(
                self.private_endpoint_config.vnet_resource_group,
                "Microsoft.Network/virtualNetworks",
                self.private_endpoint_config.vnet_name,
                self.private_endpoint_config.vnet_subscription_id)
            # if add in tags and not create_private_dns_zone
            # don't construct Private Dns Zones
            if self.tags and "create_private_dns_zone" in self.tags and not self.tags["create_private_dns_zone"]:
                return
            # construct Microsoft.Network/privateDnsZones
            private_dns_deployment_name = "PrivateDns-{}".format(str(uuid.uuid4()))
            private_dns_zones = build_private_dns_zones(
                private_dns_deployment_name,
                self.private_endpoint_config.name,
                self.private_endpoint_config.vnet_subscription_id,
                self.private_endpoint_config.vnet_resource_group,
                self.private_dns_zone_names,
                self.tags)
            self.master_template.add_resource(private_dns_zones)
            # construct Microsoft.Network/privateDnsZones/virtualNetworkLinks
            virtual_network_links = build_virtual_network_links(
                private_dns_deployment_name,
                self.private_endpoint_config.name,
                vnet_resource_id,
                self.private_endpoint_config.vnet_subscription_id,
                self.private_endpoint_config.vnet_resource_group,
                self.private_dns_zone_names,
                self.tags)
            self.master_template.add_resource(virtual_network_links)
            # construct Microsoft.Network/privateEndpoints/privateDnsZoneGroups
            private_dns_groups = build_private_dns_zone_groups(
                private_dns_deployment_name,
                self.private_endpoint_config.name,
                self.location,
                self.private_endpoint_config.vnet_subscription_id,
                self.private_endpoint_config.vnet_resource_group,
                self.private_dns_zone_names)
            self.master_template.add_resource(private_dns_groups)

    def _init_private_dns_zone_names_and_group_id(self):
        self.private_dns_zone_names = []
        self.group_ids = []
        private_link_resources = _commands.list_private_link_resources(
            self.auth,
            self.workspace_subscription_id,
            self.workspace_resource_group_name,
            self.workspace_name)
        # TODO: for now we only support one group_id - 'amlworkspace'
        # but when we support more we need to revisit the code to support
        # multiple group_ids
        private_link_resource = next(
            filter(
                lambda resource: resource.group_id == "amlworkspace",
                private_link_resources.value))
        self.group_ids.append(private_link_resource.group_id)
        for zone_name in private_link_resource.required_zone_names:
            self.private_dns_zone_names.append(zone_name)
