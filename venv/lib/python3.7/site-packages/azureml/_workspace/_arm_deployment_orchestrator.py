# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._base_sdk_common.user_agent import get_user_agent
from azureml._base_sdk_common import _ClientSessionId
from azureml._restclient.constants import RequestHeaders
from azureml._vendor.azure_resources import ResourceManagementClient


class ArmDeploymentOrchestrator(object):

    def __init__(self, auth, resource_group_name, subscription_id, deployment_name):
        self.auth = auth
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name

        self.deployment_name = deployment_name
        self.deployments_client = auth._get_service_client(ResourceManagementClient, subscription_id).deployments
        self.deployment_operations_client = auth._get_service_client(ResourceManagementClient,
                                                                     subscription_id).deployment_operations

        self.error = None

    def _arm_deploy_template(self, template):
        """
        Deploys ARM template to create a container registry.
        """
        from azureml._vendor.azure_resources.models import DeploymentProperties
        properties = DeploymentProperties(template=template, parameters={}, mode='incremental')

        headers = {
            RequestHeaders.USER_AGENT: get_user_agent(),
            RequestHeaders.CLIENT_SESSION_ID: _ClientSessionId,
            RequestHeaders.CALL_NAME: self._arm_deploy_template.__name__,
            'x-ms-client-amlsdk': None
        }
        # set the polling frequency to 2 secs so that AzurePoller polls
        # for status of our operation every two seconds rather than the default of 30 secs
        operation_config = {}
        operation_config['long_running_operation_timeout'] = 2
        self.poller = self.deployments_client.create_or_update(self.resource_group_name, self.deployment_name,
                                                               properties, custom_headers=headers,
                                                               operation_config=operation_config)

    def _check_deployment_status(self):
        try:
            headers = {
                RequestHeaders.USER_AGENT: get_user_agent(),
                RequestHeaders.CLIENT_SESSION_ID: _ClientSessionId,
                RequestHeaders.CALL_NAME: self._check_deployment_status.__name__
            }
            deployment_operations = self.deployment_operations_client.list(self.resource_group_name,
                                                                           self.deployment_name,
                                                                           custom_headers=headers)

            for deployment_operation in deployment_operations:
                properties = deployment_operation.properties
                provisioning_state = properties.provisioning_state
                target_resource = properties.target_resource

                if target_resource is None:
                    continue

                resource_name = target_resource.resource_name

                resource_type, previous_state = self.resources_being_deployed[resource_name]

                duration = ""
                try:
                    # azure-mgmt-resource=3.0.0 has duration inside properties and is an attribute
                    duration = properties.duration
                except AttributeError:
                    try:
                        # azure-mgmt-resource < 3.0.0 has duration inside additional_properties
                        # and it is a dictionary
                        duration = properties.additional_properties.get('duration', "")
                    except Exception:
                        pass

                # duration comes in format: "PT1M56.3454108S"
                try:
                    duration_parts = duration.replace("PT", "").replace("S", "").split("M")
                    if len(duration_parts) > 1:
                        duration = (float(duration_parts[0]) * 60) + float(duration_parts[1])
                    else:
                        duration = float(duration_parts[0])

                    duration = round(duration, 2)
                except Exception:
                    duration = ""
                    pass

                if provisioning_state == "Failed" and previous_state != "Failed":
                    self.resources_being_deployed[resource_name] = (resource_type, provisioning_state)
                    print("Deployment of {0} with name {1} failed.".format(resource_type, resource_name))
                # First time we're seeing this so let the user know it's being deployed
                elif properties.provisioning_state == "Running" and previous_state is None:
                    print("Deploying {0} with name {1}.".format(resource_type, resource_name))
                    self.resources_being_deployed[resource_name] = (resource_type, provisioning_state)
                # If the provisioning has already succeeded but we hadn't seen it Running before
                # (really quick deployment - so probably never happening) let user know resource
                # is being deployed and then let user know it has been deployed
                elif properties.provisioning_state == "Succeeded" and previous_state is None:
                    print("Deploying {0} with name {1}.".format(resource_type, resource_name))
                    print("Deployed {0} with name {1}. Took {2} seconds.".format(resource_type, resource_name,
                                                                                 duration))
                    self.resources_being_deployed[resource_name] = (resource_type, provisioning_state)
                # Finally, deployment has succeeded and was previously running, so mark it as finished
                elif properties.provisioning_state == "Succeeded" and previous_state != "Succeeded":
                    print("Deployed {0} with name {1}. Took {2} seconds.".format(resource_type, resource_name,
                                                                                 duration))
                    self.resources_being_deployed[resource_name] = (resource_type, provisioning_state)
        except Exception:  # if the call fails for whatever reason, the user doesn't get feedback
            pass
