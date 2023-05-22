# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import re
import subprocess

from datetime import datetime

import logging

from azureml._vendor.azure_storage.blob import ContainerSasPermissions
from azureml._vendor.azure_storage.blob import generate_container_sas

module_logger = logging.getLogger(__name__)
WASB_REGEX = r'wasbs?://(.*)@(.*)\.blob\.(core.windows.net|core.chinacloudapi.cn|core.usgovcloudapi.net)$'
WASB_MATCHES = ['.blob.core.windows.net', '.blob.core.chinacloudapi.cn', 'blob.core.usgovcloudapi.net']


def get_wasb_container_url():
    get_wasb_url_cmd = ["hdfs", "getconf", "-confKey", "fs.defaultFS"]
    return subprocess.check_output(get_wasb_url_cmd).strip().decode('utf-8')


def get_url_components(uri):
    wasb_regex = WASB_REGEX
    match = re.search(wasb_regex, uri)

    # Extract storage account name and container name from the above URL
    storage_container_name = match.group(1)
    storage_account_name = match.group(2)
    endpoint_suffix = match.group(3)
    return storage_container_name, storage_account_name, endpoint_suffix


def get_regular_container_path(wasb_container_uri):
    module_logger.debug("Remapping wasb to https: {0}".format(wasb_container_uri))

    storage_container_name, storage_account_name, endpoint_suffix = get_url_components(wasb_container_uri)
    res = "https://{0}.blob.{2}/{1}".format(
        storage_account_name, storage_container_name, endpoint_suffix)
    module_logger.debug("Mapped to {0}".format(res))
    return res


def get_container_sas(wasb_container_url=None, request_session=None):
    if (wasb_container_url is None):
        # Get the entire wasb container URL
        wasb_container_url = get_wasb_container_url()

    module_logger.debug(
        "Generating container-level Read SAS for {0}".format(wasb_container_url))

    if all(x not in wasb_container_url for x in WASB_MATCHES):
        module_logger.debug(
            "Outputs Error: Currently - Only default wasb file systems are supported to generate SAS URLs for Outputs")
        # TODO: log error or something - better handling
        return ""

    # Extract storage account, container and endpoint suffix from the above URL
    storage_container_name, storage_account_name, endpoint_suffix = get_url_components(wasb_container_url)

    # Get Encrypted Key #
    ACCOUNT_KEY_CONF_FMT = "fs.azure.account.key.{StorageAccountName}.blob.{EndPointSuffix}"
    get_hdfs_encrypted_key_cmd = ["hdfs", "getconf", "-confKey",
                                  ACCOUNT_KEY_CONF_FMT.format(StorageAccountName=storage_account_name,
                                                              EndPointSuffix=endpoint_suffix)]
    encrypted_key = subprocess.check_output(
        get_hdfs_encrypted_key_cmd).strip().decode('utf-8')

    # Get Decrypted Key #
    get_hdfs_decrypted_key_cmd = ["/usr/lib/hdinsight-common/scripts/decrypt.sh",
                                  "{encrypted_key}".format(encrypted_key=encrypted_key)]
    storage_account_key = subprocess.check_output(get_hdfs_decrypted_key_cmd)

    # Create a block blob service instance
    permissions = ContainerSasPermissions(read=True, list=True)
    container_sas = generate_container_sas(
        account_name=storage_account_name,
        container_name=storage_container_name,
        account_key=storage_account_key,
        permission=permissions,
        expiry=datetime.max)

    return "?{0}".format(container_sas)
