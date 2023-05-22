# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing an Azure Container Registry."""

import collections

from azureml._base_sdk_common.abstract_run_config_element import _AbstractRunConfigElement
from azureml._base_sdk_common.field_info import _FieldInfo


class RegistryIdentity(_AbstractRunConfigElement):
    """Registry Identity.

    :var resource_id: User assigned managed identity resource ID.
    :vartype resource_id: str

    :var client_id: User assigned managed identity client ID.
    :vartype client_id: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("resource_id", _FieldInfo(str, "User assigned managed identity resource ID")),
        ("client_id", _FieldInfo(str, "User assigned managed identity client ID"))
    ])

    def __init__(self):
        """Class ContainerRegistry constructor."""
        super(RegistryIdentity, self).__init__()
        self.resource_id = None
        self.client_id = None
        self._initialized = True


class ContainerRegistry(_AbstractRunConfigElement):
    """Defines a connection to an Azure Container Registry.

    :var address: The DNS name or IP address of the Azure Container Registry (ACR).
    :vartype address: str

    :var username: The username for ACR.
    :vartype username: str

    :var password: The password for ACR.
    :vartype password: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("address", _FieldInfo(str, "DNS name or IP address of azure container registry(ACR)")),
        ("username", _FieldInfo(str, "The username for ACR")),
        ("password", _FieldInfo(str, "The password for ACR")),
        ("registry_identity", _FieldInfo(RegistryIdentity, "RegistryIdentity")),
    ])

    def __init__(self):
        """Class ContainerRegistry constructor."""
        super(ContainerRegistry, self).__init__()
        self.address = None
        self.username = None
        self.password = None
        self.registry_identity = None
        self._initialized = True
