# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This file was created by AML team, doesn't come from azure-cli-core
# If we are updating the vendored code, please make sure this file is there
import os
import logging
from azureml._vendor.azure_cli_core.cloud import AZURE_PUBLIC_CLOUD, _convert_arm_to_cli, KNOWN_CLOUDS
from azureml._vendor.azure_cli_core._environment import get_az_config_dir

logger = logging.getLogger(__name__)
# set the timeout to 30 seconds
DEFAULT_TIMEOUT = 30
AZUREML_CLOUD_ENV_NAME = "AZUREML_CURRENT_CLOUD"


class _Clouds(object):
    """The cloud class used only for azureml"""

    _clouds = None

    @staticmethod
    def get_cloud_or_default(cloud_name):
        """Retrieves the named cloud, or Azure public cloud if the named cloud couldn't be found.

        :param cloud_name: The name of the cloud to retrieve. Can be one of AzureCloud, AzureChinaCloud, or AzureUSGovernment.
                          If no cloud is provided, if the configured default cloud from the Azure CLI is found, the default
                          will be used. If default cloud is not found, the first cloud is used.
        :type cloud_name: str
        :return: The named cloud, any configured default cloud, or the first cloud.
        :rtype: azureml._vendor.azure_cli_core.cloud.Cloud
        """
        if not cloud_name:
            cloud_name = os.getenv(AZUREML_CLOUD_ENV_NAME)
        if not _Clouds._clouds:
            existing_clouds = {KNOWN_CLOUDS[i].name: KNOWN_CLOUDS[i] for i in range(0, len(KNOWN_CLOUDS), 1)}
            if cloud_name and cloud_name in existing_clouds:
                _Clouds._clouds = existing_clouds
            else:
                _all_clouds = _Clouds._get_clouds()
                if _all_clouds:
                    _Clouds._clouds = {_all_clouds[i].name: _all_clouds[i] for i in range(0, len(_all_clouds), 1)}
                    # it seems the arm metadata has too many issues so far. Before the metadata is fixed,
                    # we force to use the hardcoded settings if azure public cloud exists in the list.
                    # for example: For US Gov cloud, endpoint resource_manager ends with slash in the hardcoded values
                    # while all other clouds don't. My hacky fix can't address this issue. In order to keep the same
                    # settings, I forced to use hardcoded settings if public cloud name is found here.
                    if AZURE_PUBLIC_CLOUD.name in _Clouds._clouds:
                        _Clouds._clouds = existing_clouds
                else:
                    _Clouds._clouds = existing_clouds

        if cloud_name in _Clouds._clouds:
            return _Clouds._clouds[cloud_name]
        else:
            default_cloud_name = _Clouds.get_default_cloud_name_from_config()
            if default_cloud_name in _Clouds._clouds:
                return _Clouds._clouds[default_cloud_name]
            else:
                return list(_Clouds._clouds.values())[0]

    @staticmethod
    def get_default_cloud_name_from_config():
        """Retrieves the name of the configured default cloud, or the name of the Azure public cloud if a default couldn't be found.

        :return: The name of the configured default cloud, or the name of the Azure public cloud if a default cloud couldn't be found.
        :rtype: str
        """
        config_dir = get_az_config_dir()
        config_path = os.path.join(config_dir, 'config')
        TARGET_CONFIG_SECTION_LITERAL = '[cloud]'
        TARGET_CONFIG_KEY_LITERAL = 'name = '
        cloud = AZURE_PUBLIC_CLOUD.name
        foundCloudSection = False
        try:
            with open(config_path, 'r') as f:
                line = f.readline()
                while line != '':
                    if line.startswith(TARGET_CONFIG_SECTION_LITERAL):
                        foundCloudSection = True

                    if foundCloudSection:
                        if line.startswith(TARGET_CONFIG_KEY_LITERAL):
                            cloud = line[len(TARGET_CONFIG_KEY_LITERAL):].strip()
                            break
                        if line.strip() == '':
                            break

                    line = f.readline()
        except IOError:
            pass

        return cloud

    @staticmethod
    def _get_clouds_by_metadata_url(metadata_url, timeout=DEFAULT_TIMEOUT):
        """Get all the clouds by the specified metadata url

            :return: list of the clouds
            :rtype: list[azureml._vendor.azure_cli_core.Cloud]
        """
        try:
            import requests
            logger.debug('Start : Loading cloud metatdata from the url specified by {0}'.format(metadata_url))
            with requests.get(metadata_url, timeout=timeout) as meta_response:
                arm_cloud_dict = meta_response.json()
                cli_cloud_dict = _convert_arm_to_cli(arm_cloud_dict)
                # Strip trailing slash at end of active directory endpoints
                for cloud in cli_cloud_dict.keys():
                    logger.debug('Active directory endpoint loaded from {0} metadata is {1}'.format(cloud, cli_cloud_dict[cloud].endpoints.active_directory))
                    cli_cloud_dict[cloud].endpoints.active_directory = cli_cloud_dict[cloud].endpoints.active_directory.rstrip('/')
                    logger.debug('Active directory endpoint for cloud {0} set to {1}'.format(cloud, cli_cloud_dict[cloud].endpoints.active_directory))
                logger.debug('Finish : Loading cloud metatdata from the url specified by {0}'.format(metadata_url))
                return list(cli_cloud_dict.values())
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning("Error: Azure ML was unable to load cloud metadata from the url specified by {0}. {1}. "
                           "This may be due to a misconfiguration of networking controls. Azure Machine Learning Python SDK "
                           "requires outbound access to Azure Resource Manager. Please contact your networking team to configure "
                           "outbound access to Azure Resource Manager on both Network Security Group and Firewall. "
                           "For more details on required configurations, see "
                           "https://docs.microsoft.com/azure/machine-learning/how-to-access-azureml-behind-firewall.".format(
                metadata_url, ex))

    @staticmethod
    def _get_clouds():
        """Get all the clouds from metadata url list

            :return: list of the clouds
            :rtype: list[azureml._vendor.azure_cli_core.Cloud]
        """
        metadata_url_list = [
            "https://management.azure.com/metadata/endpoints?api-version=2019-05-01"]
        clouds = []
        # Iterate the metadata_url_list, if any one returns non-empty list, return it
        logger.debug('Start : Loading cloud metatdata')
        for metadata_url in metadata_url_list:
            all_clouds = _Clouds._get_clouds_by_metadata_url(metadata_url)
            if all_clouds:
                logger.debug('Finish : Loading cloud metatdata')
                return all_clouds
        logger.debug('Finish : Loading cloud metatdata')
        return clouds
